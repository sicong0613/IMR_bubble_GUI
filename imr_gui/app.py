from __future__ import annotations

import functools
import importlib
import csv
import json
import sys
import traceback
from dataclasses import asdict, dataclass
from io import BytesIO
import time
import math
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import medfilt
try:
    import mat73 as _mat73
    _HAS_MAT73 = True
except ImportError:
    _HAS_MAT73 = False
from PySide6.QtCore import QByteArray, QEvent, QMimeData, Qt, QThread, Signal, QTimer
from PySide6.QtGui import QActionGroup, QColor, QImage, QPen, QValidator
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QSizePolicy,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressDialog,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from imr_gui.imr import NhkvInputs, NhkvOutputs, simulate_nhkv_lic
from imr_gui.imr import NhkvRmaxInputs, simulate_nhkv_rmax_lic
from imr_gui.imr import GMODInputs, simulate_gmod_lic
from imr_gui.imr import GMOD1Inputs, simulate_gmod1_lic
from imr_gui.io import load_experiment_mat, find_rmax_value
from imr_gui.io.mat_loader import ExperimentData
from imr_gui.io.mat_loader import _find_rmax_time
from imr_gui.ui.mpl_canvas import MplCanvas, PlotHandles
from imr_gui.constitutive import (
    load_nhkv_model,
    ConstitutiveParameter, ConstitutiveModel, load_available_models,
)
from imr_gui.opt import (
    OptConfig, FitConfig, FitProgress, FitResult,
    fit_nhkv_to_experiment,
    AVAILABLE_METHODS, DE_STRATEGIES, _HAS_CMA,
)


# ---------------------------------------------------------------------------
# spin-box helpers
# ---------------------------------------------------------------------------


class _NoWheelSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores mouse-wheel events so that scrolling
    the parent QScrollArea works instead of changing the value."""

    def wheelEvent(self, event):  # noqa: N802
        event.ignore()


class _NoWheelComboBox(QComboBox):
    """QComboBox that ignores mouse-wheel events (prevents accidental
    value changes while scrolling the parent panel)."""

    def wheelEvent(self, event):  # noqa: N802
        event.ignore()


class _MatDropFrame(QFrame):
    """Small drag/drop target for MAT files."""

    fileDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("matDropFrame")
        self.setStyleSheet(
            "#matDropFrame {"
            "border: 2px dashed #7a7a7a;"
            "border-radius: 4px;"
            "background: rgba(255, 255, 255, 0.03);"
            "}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        label = QLabel("Drop .mat file here")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(label)

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(".mat"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):  # noqa: N802
        for url in event.mimeData().urls():
            if url.isLocalFile() and url.toLocalFile().lower().endswith(".mat"):
                self.fileDropped.emit(url.toLocalFile())
                event.acceptProposedAction()
                return
        event.ignore()


class _SigFigSpinBox(_NoWheelSpinBox):
    """QDoubleSpinBox that displays 4 significant figures, automatically using
    scientific notation for very large or very small values."""

    def textFromValue(self, val: float) -> str:  # noqa: N802
        if val == 0.0:
            return "0.000"
        mag = abs(val)
        if mag < 1e-3 or mag >= 1e4:
            return f"{val:.3e}"
        if mag >= 1.0:
            digits_before = int(math.floor(math.log10(mag))) + 1
            dec = max(0, 4 - digits_before)
        else:
            dec = int(-math.floor(math.log10(mag))) + 3
        return f"{val:.{dec}f}"

    def valueFromText(self, text: str) -> float:  # noqa: N802
        try:
            return float(text.strip())
        except ValueError:
            return 0.0

    def validate(self, text: str, pos: int):  # noqa: N802
        t = text.strip()
        if not t or t in ("-", "+", ".", "-."):
            return QValidator.State.Intermediate, text, pos
        try:
            val = float(t)
            if self.minimum() <= val <= self.maximum():
                return QValidator.State.Acceptable, text, pos
            return QValidator.State.Intermediate, text, pos
        except ValueError:
            pass
        # Allow partial scientific notation like "1e", "1e-", "1.5e+"
        if re.fullmatch(r'[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d*', t) or \
           re.fullmatch(r'[-+]?(\d+\.?\d*|\.\d+)[eE]?', t):
            return QValidator.State.Intermediate, text, pos
        return QValidator.State.Invalid, text, pos


class _SciNotationSpinBox:
    """Wrapper that makes a QDoubleSpinBox display/accept scientific notation
    like ``1e-7`` instead of ``0.0000001``.  Works by intercepting the
    QLineEdit and syncing manually."""

    def __init__(self, spin: QDoubleSpinBox, initial: float):
        self._spin = spin
        self._value = initial
        le = spin.lineEdit()
        le.setText(f"{initial:.0e}")
        le.editingFinished.connect(self._on_edited)

    def _on_edited(self):
        txt = self._spin.lineEdit().text().strip()
        try:
            v = float(txt)
            if v > 0:
                self._value = v
        except ValueError:
            pass
        self._spin.lineEdit().setText(f"{self._value:.0e}")

    def value(self) -> float:
        return self._value


class _JobRowDelegate(QStyledItemDelegate):
    """Paint selected job rows with an outer border while preserving status color."""

    def paint(self, painter, option, index):  # noqa: N802
        opt = QStyleOptionViewItem(option)
        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        if selected:
            opt.state &= ~QStyle.StateFlag.State_Selected

        super().paint(painter, opt, index)

        if not selected:
            return

        painter.save()
        pen = QPen(QColor(61, 120, 216))
        pen.setWidth(2)
        painter.setPen(pen)

        rect = option.rect.adjusted(1, 1, -1, -1)
        col = index.column()
        last_col = index.model().columnCount() - 1

        painter.drawLine(rect.left(), rect.top(), rect.right(), rect.top())
        painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())
        if col == 0:
            painter.drawLine(rect.left(), rect.top(), rect.left(), rect.bottom())
        if col == last_col:
            painter.drawLine(rect.right(), rect.top(), rect.right(), rect.bottom())
        painter.restore()


class _ColumnResizeHandle(QFrame):
    dragged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(5)
        self.setFixedHeight(24)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.SplitHCursor)
        self.setFrameShape(QFrame.Shape.VLine)
        self.setStyleSheet("QFrame { color: #3f3f3f; }")
        self._last_x: int | None = None

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_x = int(event.globalPosition().x())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._last_x is None:
            super().mouseMoveEvent(event)
            return
        x = int(event.globalPosition().x())
        delta = x - self._last_x
        if delta:
            self.dragged.emit(delta)
            self._last_x = x
        event.accept()

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._last_x = None
        event.accept()


class _ColorSwatchDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):  # noqa: N802
        data = index.data(Qt.ItemDataRole.UserRole)
        if data == "__more__":
            super().paint(painter, option, index)
            return
        color = QColor(str(data))
        if not color.isValid():
            super().paint(painter, option, index)
            return
        painter.save()
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        rect = option.rect.adjusted(4, 3, -4, -3)
        painter.fillRect(rect, color)
        painter.setPen(QPen(QColor("#000000"), 1))
        painter.drawRect(rect)
        painter.restore()

    def sizeHint(self, option, index):  # noqa: N802
        size = super().sizeHint(option, index)
        data = index.data(Qt.ItemDataRole.UserRole)
        if data != "__more__":
            size.setHeight(max(size.height(), 24))
        return size


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


@dataclass
class AppState:
    exp_t: np.ndarray | None = None
    exp_R: np.ndarray | None = None
    exp_path: str | None = None
    import_metadata: dict | None = None
    view_mode: str = "dimensional"  # or "normalized"
    sim_t: np.ndarray | None = None
    sim_R: np.ndarray | None = None
    sim_meta: dict | None = None  # {"Rmax": float, "t_rmax": float, "tc": float}
    sim_out: NhkvOutputs | None = None  # full solver output for export
    P_inf: float | None = None
    rho: float | None = None
    R_eq: float | None = None
    param_bounds: dict[str, dict] | None = None
    mode: str = "simulation"  # "simulation" | "fitting" | "jobs"
    best_fit_t: np.ndarray | None = None
    best_fit_R: np.ndarray | None = None
    best_fit_meta: dict | None = None


# ---------------------------------------------------------------------------
# picklable simulation specification (for DE multiprocessing)
# ---------------------------------------------------------------------------


@dataclass
class _SimSpec:
    """All data needed to run one simulation — fully picklable (no Qt refs)."""
    model_key: str
    Req: float
    NT: int
    P_inf: float
    rho: float
    const: dict
    solver: dict
    Rmax_exp: float = 0.0  # used only by NHKV (Rmax); 0 means fall back to Req
    bubble_model: str = "Keller-Miksis"
    plugin_entrypoint: str = ""
    plugin_context: dict | None = None


@dataclass
class _PluginSimSpec:
    entrypoint: str
    context: dict


def _sim_spec_call(spec: _SimSpec, params_si: dict, tspan: float):
    """Module-level dispatch function — picklable for ProcessPoolExecutor workers."""
    key = spec.model_key
    bm = spec.bubble_model
    if key == "NHKV":
        const_kw = {k: v for k, v in spec.const.items()
                    if k in NhkvInputs.__dataclass_fields__}
        return simulate_nhkv_lic(NhkvInputs(
            U0=params_si["U0"], G=params_si["G"], mu=params_si["mu"],
            Req=spec.Req, tspan=tspan, NT=spec.NT,
            P_inf=spec.P_inf, rho=spec.rho, bubble_model=bm, **spec.solver, **const_kw,
        ))
    elif key == "NHKV (Rmax)":
        Rmax_exp = spec.Rmax_exp if spec.Rmax_exp > 0 else spec.Req
        const_kw = {k: v for k, v in spec.const.items()
                    if k in NhkvRmaxInputs.__dataclass_fields__}
        return simulate_nhkv_rmax_lic(NhkvRmaxInputs(
            G=params_si["G"], mu=params_si["mu"],
            Req=spec.Req, Rmax_exp=Rmax_exp, tspan=tspan, NT=spec.NT,
            P_inf=spec.P_inf, rho=spec.rho, bubble_model=bm, **spec.solver, **const_kw,
        ))
    elif key == "GMOD1":
        const_kw = {k: v for k, v in spec.const.items()
                    if k in GMOD1Inputs.__dataclass_fields__}
        return simulate_gmod1_lic(GMOD1Inputs(
            U0=params_si.get("U0", 100.0),
            GA=params_si.get("GA", 8e6),
            alpha=params_si.get("alpha", 1.0),
            GB=params_si.get("GB", 1e4),
            beta=params_si.get("beta", 1.0),
            mu=params_si.get("mu", 0.226),
            lambda_Y=params_si.get("lambda_Y", 1.5),
            Req=spec.Req, tspan=tspan, NT=spec.NT,
            P_inf=spec.P_inf, rho=spec.rho, bubble_model=bm, **spec.solver, **const_kw,
        ))
    elif key == "GMOD2":
        const_kw = {k: v for k, v in spec.const.items()
                    if k in GMODInputs.__dataclass_fields__}
        return simulate_gmod_lic(GMODInputs(
            U0=params_si.get("U0", 100.0),
            GA1=params_si.get("GA1", 8e6),
            GA2=params_si.get("GA2", 1e-10),
            alpha1=params_si.get("alpha1", 1.0),
            alpha2=params_si.get("alpha2", 1.0),
            GB1=params_si.get("GB1", 1e4),
            GB2=params_si.get("GB2", 1e-10),
            beta1=params_si.get("beta1", 1.0),
            beta2=params_si.get("beta2", 1.0),
            mu=params_si.get("mu", 0.226),
            lambda_Y=params_si.get("lambda_Y", 1.5),
            Req=spec.Req, tspan=tspan, NT=spec.NT,
            P_inf=spec.P_inf, rho=spec.rho, bubble_model=bm, **spec.solver, **const_kw,
        ))
    if spec.plugin_entrypoint:
        ctx = dict(spec.plugin_context or {})
        ctx.update({
            "Req": spec.Req,
            "NT": spec.NT,
            "P_inf": spec.P_inf,
            "rho": spec.rho,
            "gamma": float(spec.const.get("gamma", 0.056)),
            "c_long": float(spec.const.get("c_long", 1485.0)),
            "bubble_model": bm,
            "constants": dict(spec.const),
            "solver": dict(spec.solver),
            "Rmax_exp": spec.Rmax_exp,
            "model_key": key,
        })
        return _plugin_sim_call(_PluginSimSpec(spec.plugin_entrypoint, ctx), params_si, tspan)
    raise ValueError(f"No solver is registered for model '{key}'.")


def _load_entrypoint(entrypoint: str):
    if ":" not in entrypoint:
        raise ValueError("solver_entrypoint must have the form 'module:function'.")
    module_name, func_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name)
    if not callable(fn):
        raise TypeError(f"Solver entrypoint is not callable: {entrypoint}")
    return fn


def _plugin_sim_call(spec: _PluginSimSpec, params_si: dict, tspan: float):
    fn = _load_entrypoint(spec.entrypoint)
    context = dict(spec.context)
    context["tspan"] = float(tspan)
    return fn(dict(params_si), float(tspan), context)


def _run_plugin_simulation(inp: dict):
    return _plugin_sim_call(
        _PluginSimSpec(inp["solver_entrypoint"], inp["context"]),
        inp["params_si"],
        float(inp["tspan"]),
    )


def _run_sim_spec_payload(inp: dict):
    return _sim_spec_call(
        inp["spec"],
        inp["params_si"],
        float(inp["tspan"]),
    )


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------


class SimWorker(QThread):
    finished_ok = Signal(object)
    failed = Signal(str, str)

    def __init__(self, simulate_fn, inputs, parent=None):
        super().__init__(parent)
        self._fn = simulate_fn
        self._inputs = inputs

    def run(self):
        import traceback as _tb
        try:
            out = self._fn(self._inputs)
            self.finished_ok.emit(out)
        except Exception as e:
            self.failed.emit(str(e), _tb.format_exc())


class FitWorker(QThread):
    finished_ok = Signal(object)
    failed = Signal(str, str)
    progress = Signal(object)

    def __init__(self, cfg, bounds_si, fit_flags, scales, initial_values,
                 opt_config: "OptConfig | None" = None, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._bounds_si = bounds_si
        self._fit_flags = fit_flags
        self._scales = scales
        self._initial_values = initial_values
        self._opt_config = opt_config or OptConfig()
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        import traceback as _tb
        try:
            res = fit_nhkv_to_experiment(
                self._cfg,
                self._bounds_si,
                fit_flags=self._fit_flags,
                scales=self._scales,
                initial_values=self._initial_values,
                progress_callback=lambda p: self.progress.emit(p),
                stop_flag=lambda: self._stop_requested,
                opt_config=self._opt_config,
            )
            self.finished_ok.emit(res)
        except Exception as e:
            self.failed.emit(str(e), _tb.format_exc())


# ---------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    APP_TITLE = "IMR Fitting GUI (beta 1.2)"
    PARAM_INHERIT_ALIASES: dict[str, dict[str, str]] = {
        "GMOD1": {
            "GA": "GA1",
            "alpha": "alpha1",
            "GB": "GB1",
            "beta": "beta1",
        },
        "GMOD2": {
            "GA1": "GA",
            "alpha1": "alpha",
            "GB1": "GB",
            "beta1": "beta",
        },
    }
    DEFAULT_IMPORT_WIZARD_KEYWORDS = {
        "t_exp": ["t_exp", "t", "time", "time_exp"],
        "R_exp": ["R_exp", "R", "radius", "radius_exp", "R1_exp"],
        "t_sim": ["t_sim", "simulation_t", "sim_t", "t_fit", "best_fit_t"],
        "R_sim": ["R_sim", "simulation_R", "sim_R", "R_fit", "best_fit_R"],
        "legend": ["legend", "label", "curve_label"],
        "Req": ["Req", "R_eq", "R_equilibrium", "R1_eq"],
        "Rmax": ["Rmax", "R_max"],
        "P_inf": ["P_inf", "Pinf", "pinf"],
        "rho": ["rho", "density"],
        "c_long": ["c_long", "c", "sound_speed"],
        "gamma": ["gamma", "surface_tension"],
        "t_start": ["t_start", "fit_window_start"],
        "t_end": ["t_end", "fit_window_end"],
        "LSQErr": ["LSQErr", "lsqerr", "loss"],
    }
    DEFAULT_IMPORT_WIZARD_UNITS = {
        "t_exp": "s",
        "R_exp": "m",
        "t_sim": "s",
        "R_sim": "m",
        "Req": "m",
        "Rmax": "m",
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.APP_TITLE)
        self.resize(1200, 700)

        self.state = AppState()
        self._sim_worker: SimWorker | None = None
        self._sim_dialog: QProgressDialog | None = None
        self._sim_timer: QTimer = QTimer(self)
        self._sim_timer.setInterval(200)
        self._sim_timer.timeout.connect(self._update_sim_progress)
        self._sim_start_time: float | None = None
        self._last_sim_duration: float | None = None
        self._fit_worker: FitWorker | None = None
        self._fit_dialog: QProgressDialog | None = None
        self._fit_timer: QTimer = QTimer(self)
        self._fit_timer.setInterval(500)
        self._fit_timer.timeout.connect(self._update_fit_progress)
        self._fit_start_time: float | None = None
        self._queue_running: bool = False
        self._queue_stop_after_current: bool = False
        self._queue_output_dir: Path | None = None
        self._queue_current_index: int | None = None
        self._queue_fit_worker: FitWorker | None = None
        self._queue_fit_workers: dict[int, FitWorker] = {}
        self._queue_sim_workers: dict[int, SimWorker] = {}
        self._queue_export_names: set[str] = set()
        self._opt_config: OptConfig = OptConfig()
        self._job_success_threshold_enabled: bool = True
        self._job_success_lsqerr: float = 10.0
        self._job_output_dir: str = ""
        self._job_ask_output_dir: bool = True
        self._job_chain_best_fit_initial: bool = False
        self._job_parallel_workers: int = 1
        self._job_import_name_cleanup_enabled: bool = True
        self._job_import_name_cleanup_regex: str = ""
        self._queue_previous_seed: dict | None = None

        # model state
        self._current_model: ConstitutiveModel | None = None
        self._model_constants: dict = {}
        self._param_rows: dict[str, dict] = {}
        self._fit_widgets: dict[str, dict] = {}
        self._saved_ui_defaults: dict = {}
        self._jobs: list[dict] = []
        self._editing_job_index: int | None = None
        self._loading_job_to_editor: bool = False
        self._model_param_memory: dict[str, dict] = {}
        self._available_models = load_available_models()
        self._multi_curve_enabled: bool = False
        self._view_curves: list[dict] = []
        self._view_color_presets: list[dict] = self._load_view_color_presets()
        self._curve_color_mode: str = "distinct"
        self._selected_curve_indices: set[int] = set()
        self._curve_selection_anchor: int | None = None
        self._import_wizard_keywords: dict[str, list[str]] = self._normalise_import_wizard_keywords({})
        self._import_wizard_units: dict[str, str] = dict(self.DEFAULT_IMPORT_WIZARD_UNITS)
        self._import_wizard_um_per_pixel: float = 3.2
        self._import_wizard_fps: float = 1_000_000.0
        self._import_wizard_remove_below: bool = False
        self._import_wizard_remove_spikes: bool = False
        self._import_wizard_spike_threshold: float = 2.0

        self._build_menu()
        self._build_ui()
        self._build_curve_view_dock()
        self._load_settings()
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")
        self._set_mode("simulation")

    # =====================================================================
    # menu
    # =====================================================================

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        self._act_load_exp = file_menu.addAction("Load experiment data (.mat)")
        self._act_load_exp.triggered.connect(self.on_load_experiment)
        self._act_import_wizard = file_menu.addAction("Import Wizard...")
        self._act_import_wizard.triggered.connect(self.on_import_wizard)
        file_menu.addSeparator()
        self._act_load_params = file_menu.addAction("Load parameters (MAT)...")
        self._act_load_params.triggered.connect(self.on_load_params)
        self._act_save_params = file_menu.addAction("Save parameters (MAT)...")
        self._act_save_params.triggered.connect(self.on_save_params)
        file_menu.addSeparator()
        self._act_export_result = file_menu.addAction("Export result (.mat)...")
        self._act_export_result.triggered.connect(self.on_export_result)
        file_menu.addSeparator()
        self._act_import_jobs = file_menu.addAction("Import job queue (.imrqueue)...")
        self._act_import_jobs.triggered.connect(self.on_import_job_queue)
        self._act_export_jobs = file_menu.addAction("Export job queue (.imrqueue)...")
        self._act_export_jobs.triggered.connect(self.on_export_job_queue)
        self._act_export_jobs_csv = file_menu.addAction("Export job queue (.csv)...")
        self._act_export_jobs_csv.triggered.connect(self.on_export_job_queue_csv)
        self._act_batch_add_jobs = file_menu.addAction("Batch add experiments as jobs...")
        self._act_batch_add_jobs.triggered.connect(self.on_batch_add_experiments_as_jobs)
        self._act_batch_create_sim_jobs = file_menu.addAction("Batch create simulation jobs...")
        self._act_batch_create_sim_jobs.triggered.connect(self.on_batch_create_simulation_jobs)

        module_menu = self.menuBar().addMenu("Module")
        self._act_sim = module_menu.addAction("Simulation")
        self._act_fit_mode = module_menu.addAction("Fitting")
        self._act_jobs_mode = module_menu.addAction("Job List")
        self._act_sim.setCheckable(True)
        self._act_fit_mode.setCheckable(True)
        self._act_jobs_mode.setCheckable(True)
        self._act_sim.setChecked(True)
        ag = QActionGroup(self)
        ag.addAction(self._act_sim)
        ag.addAction(self._act_fit_mode)
        ag.addAction(self._act_jobs_mode)
        ag.setExclusive(True)
        self._act_sim.triggered.connect(lambda: self._set_mode("simulation"))
        self._act_fit_mode.triggered.connect(lambda: self._set_mode("fitting"))
        self._act_jobs_mode.triggered.connect(lambda: self._set_mode("jobs"))

        physics_menu = self.menuBar().addMenu("Physics")
        bubble_menu = physics_menu.addMenu("Bubble dynamics")
        ag_bubble = QActionGroup(self)
        ag_bubble.setExclusive(True)
        self._act_km = bubble_menu.addAction("Keller-Miksis")
        self._act_rp = bubble_menu.addAction("Rayleigh-Plesset")
        for act in (self._act_km, self._act_rp):
            act.setCheckable(True)
            ag_bubble.addAction(act)
        self._act_km.setChecked(True)
        self._bubble_model = "Keller-Miksis"
        self._act_km.triggered.connect(lambda: setattr(self, "_bubble_model", "Keller-Miksis"))
        self._act_rp.triggered.connect(lambda: setattr(self, "_bubble_model", "Rayleigh-Plesset"))
        act_phys_settings = physics_menu.addAction("Physics Settings...")
        act_phys_settings.triggered.connect(self._show_physics_settings)

        settings_menu = self.menuBar().addMenu("Settings")
        act_save_default = settings_menu.addAction("Save Current as Default")
        act_save_default.triggered.connect(self.on_save_defaults)
        act_opt_settings = settings_menu.addAction("Optimizer Settings...")
        act_opt_settings.triggered.connect(self._show_optimizer_settings)
        act_job_settings = settings_menu.addAction("Job List Settings...")
        act_job_settings.triggered.connect(self._show_job_settings)

        view_menu = self.menuBar().addMenu("View")
        self._act_curve_view = view_menu.addAction("Curve Selection Panel")
        self._act_curve_view.setCheckable(True)
        self._act_curve_view.triggered.connect(self._toggle_curve_view_panel)
        view_menu.addSeparator()
        self._act_export_view_mat = view_menu.addAction("Export view (.mat)")
        self._act_export_view_mat.triggered.connect(self._on_export_view_mat)
        self._act_export_view_svg = view_menu.addAction("Export view (.svg)")
        self._act_export_view_svg.triggered.connect(self._on_export_view_svg)
        self._act_copy_view_png = view_menu.addAction("Copy view (png)")
        self._act_copy_view_png.triggered.connect(self._on_copy_view_png)
        self._act_copy_view_svg = view_menu.addAction("Copy view (svg)")
        self._act_copy_view_svg.triggered.connect(self._on_copy_view_svg)

    # =====================================================================
    # UI build
    # =====================================================================

    def _build_ui(self):
        root = QWidget(self)
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Horizontal, root)

        # ---- left panel: editor / job list ------------------------------
        left = QStackedWidget()
        self._left_stack = left

        left_editor = QWidget()
        left_layout = QVBoxLayout(left_editor)

        # ---- Model selector row ----
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._cmb_model = _NoWheelComboBox()
        for key in self._available_models:
            self._cmb_model.addItem(key)
        self._cmb_model.setCurrentText("NHKV")
        model_row.addWidget(self._cmb_model, stretch=1)
        left_layout.addLayout(model_row)

        # ---- Build physics settings (kept as persistent widgets) ----
        self._build_physics_settings()

        # ---- parameter box with scroll area ----
        self._param_box = QGroupBox("Parameters")
        param_box_outer = QVBoxLayout(self._param_box)
        self._param_scroll = QScrollArea()
        self._param_scroll.setWidgetResizable(True)
        self._param_scroll_content = QWidget()
        self._param_layout = QVBoxLayout(self._param_scroll_content)
        self._param_layout.setContentsMargins(2, 2, 2, 2)
        self._param_scroll.setWidget(self._param_scroll_content)
        param_box_outer.addWidget(self._param_scroll)
        self._param_scroll.setMinimumHeight(120)
        left_layout.addWidget(self._param_box, stretch=1)

        # ---- Experiment box (Req + tspan only) ----
        exp_box = QGroupBox("Experiment")
        exp_lay = QVBoxLayout(exp_box)

        row_req = QHBoxLayout()
        row_req.addWidget(QLabel("Req (µm)"))
        self.spin_Req_um = _NoWheelSpinBox()
        self.spin_Req_um.setRange(0.001, 1e6)
        self.spin_Req_um.setDecimals(6)
        self.spin_Req_um.setValue(30.0)
        self.spin_Req_um.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        row_req.addWidget(self.spin_Req_um, stretch=1)
        exp_lay.addLayout(row_req)

        row_tspan = QHBoxLayout()
        row_tspan.addWidget(QLabel("tspan (µs)"))
        self.spin_tspan_us = _NoWheelSpinBox()
        self.spin_tspan_us.setRange(0.001, 1e9)
        self.spin_tspan_us.setDecimals(3)
        self.spin_tspan_us.setValue(100.0)
        self.spin_tspan_us.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        row_tspan.addWidget(self.spin_tspan_us, stretch=1)
        exp_lay.addLayout(row_tspan)

        left_layout.addWidget(exp_box, stretch=0)

        action_grid = QGridLayout()
        action_grid.setContentsMargins(0, 0, 0, 0)
        action_grid.setHorizontalSpacing(6)
        action_grid.setVerticalSpacing(6)

        self.btn_primary_action = QPushButton("Simulate")
        self.btn_primary_action.clicked.connect(self._on_primary_action)
        action_grid.addWidget(self.btn_primary_action, 0, 0)

        self.btn_add_job = QPushButton("Add to job list")
        self.btn_add_job.setEnabled(False)
        self.btn_add_job.setToolTip("Add the current fitting setup as a queued job.")
        self.btn_add_job.clicked.connect(self.on_add_job)
        action_grid.addWidget(self.btn_add_job, 0, 1)

        self.btn_save_default = QPushButton("Save as default")
        self.btn_save_default.clicked.connect(self.on_save_defaults)
        action_grid.addWidget(self.btn_save_default, 1, 0)

        self.btn_find_initial = QPushButton("Find initial")
        self.btn_find_initial.setEnabled(False)
        self.btn_find_initial.setToolTip("Initial-guess search will be added in a later version.")
        action_grid.addWidget(self.btn_find_initial, 1, 1)

        left_layout.addLayout(action_grid)

        self._job_page = QWidget()
        job_layout = QVBoxLayout(self._job_page)
        job_layout.setContentsMargins(6, 6, 6, 6)

        job_header = QWidget()
        job_header_lay = QHBoxLayout(job_header)
        job_header_lay.setContentsMargins(0, 0, 0, 0)
        job_header_lay.addWidget(QLabel("Job List"))
        job_header_lay.addStretch(1)
        self.btn_run_jobs = QPushButton("Run queue")
        self.btn_run_jobs.setEnabled(False)
        self.btn_run_jobs.clicked.connect(self.on_run_queue)
        self.btn_stop_jobs = QPushButton("Stop queue")
        self.btn_stop_jobs.setEnabled(False)
        self.btn_stop_jobs.clicked.connect(self.on_stop_queue_after_current)
        job_header_lay.addWidget(self.btn_run_jobs)
        job_header_lay.addWidget(self.btn_stop_jobs)
        job_layout.addWidget(job_header)

        self.tbl_jobs = QTableWidget(0, 4)
        self.tbl_jobs.setHorizontalHeaderLabels([
            "#", "Type", "Experiment", "Model"
        ])
        self.tbl_jobs.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_jobs.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tbl_jobs.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_jobs.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tbl_jobs.verticalHeader().setVisible(False)
        self.tbl_jobs.setShowGrid(False)
        self.tbl_jobs.setStyleSheet("QTableWidget { outline: 0; }")
        self.tbl_jobs.setItemDelegate(_JobRowDelegate(self.tbl_jobs))
        header = self.tbl_jobs.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.tbl_jobs.setColumnWidth(0, 36)
        self.tbl_jobs.setColumnWidth(1, 58)
        self.tbl_jobs.setColumnWidth(3, 72)
        job_layout.addWidget(self.tbl_jobs, stretch=1)

        job_actions = QVBoxLayout()
        job_actions_primary = QHBoxLayout()
        job_actions_reorder = QHBoxLayout()

        self.btn_job_top = QPushButton("Move to top")
        self.btn_job_top.setEnabled(False)
        self.btn_job_top.clicked.connect(self.on_move_job_to_top)
        self.btn_job_up = QPushButton("Move up")
        self.btn_job_up.setEnabled(False)
        self.btn_job_up.clicked.connect(self.on_move_job_up)
        self.btn_job_down = QPushButton("Move down")
        self.btn_job_down.setEnabled(False)
        self.btn_job_down.clicked.connect(self.on_move_job_down)
        self.btn_remove_job = QPushButton("Remove selected")
        self.btn_remove_job.setEnabled(False)
        self.btn_remove_job.clicked.connect(self.on_remove_selected_job)
        self.btn_load_job = QPushButton("Load to editor")
        self.btn_load_job.setEnabled(False)
        self.btn_load_job.clicked.connect(self.on_load_selected_job_to_editor)
        self.btn_clear_completed_jobs = QPushButton("Clear completed")
        self.btn_clear_completed_jobs.setEnabled(False)
        self.btn_clear_completed_jobs.clicked.connect(self.on_clear_completed_jobs)

        job_actions_primary.addWidget(self.btn_remove_job)
        job_actions_primary.addWidget(self.btn_load_job)
        job_actions_primary.addWidget(self.btn_clear_completed_jobs)
        job_actions_reorder.addWidget(self.btn_job_top)
        job_actions_reorder.addWidget(self.btn_job_up)
        job_actions_reorder.addWidget(self.btn_job_down)
        job_actions.addLayout(job_actions_primary)
        job_actions.addLayout(job_actions_reorder)
        job_layout.addLayout(job_actions)
        self.tbl_jobs.itemSelectionChanged.connect(self._on_job_selection_changed)

        left.addWidget(left_editor)
        left.addWidget(self._job_page)

        # ---- right panel: preview + outputs ------------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)

        preview_header = QWidget()
        ph_lay = QHBoxLayout(preview_header)
        ph_lay.setContentsMargins(0, 0, 0, 0)
        ph_lay.addWidget(QLabel("Preview"))
        ph_lay.addStretch(1)
        self.btn_dim = QPushButton("Dimensional")
        self.btn_norm = QPushButton("Normalized")
        self.btn_dim.clicked.connect(lambda: self.set_view_mode("dimensional"))
        self.btn_norm.clicked.connect(lambda: self.set_view_mode("normalized"))
        ph_lay.addWidget(self.btn_dim)
        ph_lay.addWidget(self.btn_norm)
        right_layout.addWidget(preview_header)

        self._fit_window_widget = QWidget()
        fw_lay = QHBoxLayout(self._fit_window_widget)
        fw_lay.setContentsMargins(0, 0, 0, 0)
        fw_lay.addWidget(QLabel("Fit window:"))
        self.chk_fit_window_cycles = QCheckBox("Auto")
        self.chk_fit_window_cycles.setToolTip(
            "Automatically set the fitting window from Rmax to the selected "
            "collapse minimum."
        )
        fw_lay.addWidget(self.chk_fit_window_cycles)
        self.spin_fit_window_cycles = QSpinBox()
        self.spin_fit_window_cycles.setRange(0, 20)
        self.spin_fit_window_cycles.setValue(1)
        self.spin_fit_window_cycles.setToolTip("Number of collapse cycles to include.")
        fw_lay.addWidget(self.spin_fit_window_cycles)
        fw_lay.addWidget(QLabel("cycles"))
        fw_lay.addWidget(QLabel("t_start (µs)"))
        self.spin_t_fit_start = _NoWheelSpinBox()
        self.spin_t_fit_start.setRange(-1e9, 1e9)
        self.spin_t_fit_start.setDecimals(3)
        self.spin_t_fit_start.setValue(0.0)
        self.spin_t_fit_start.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        fw_lay.addWidget(self.spin_t_fit_start, stretch=1)
        fw_lay.addWidget(QLabel("t_end (µs)"))
        self.spin_t_fit_end = _NoWheelSpinBox()
        self.spin_t_fit_end.setRange(-1e9, 1e9)
        self.spin_t_fit_end.setDecimals(3)
        self.spin_t_fit_end.setValue(100.0)
        self.spin_t_fit_end.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        fw_lay.addWidget(self.spin_t_fit_end, stretch=1)
        right_layout.addWidget(self._fit_window_widget)
        self.spin_t_fit_start.valueChanged.connect(self._on_fit_window_changed)
        self.spin_t_fit_end.valueChanged.connect(self._on_fit_window_changed)
        self.chk_fit_window_cycles.toggled.connect(self._on_fit_window_auto_changed)
        self.spin_fit_window_cycles.valueChanged.connect(self._on_fit_window_cycles_changed)

        # Canvas wrapped with zoom sliders
        canvas_area = QWidget()
        canvas_grid = QGridLayout(canvas_area)
        canvas_grid.setContentsMargins(0, 0, 0, 0)
        canvas_grid.setSpacing(2)

        self.slider_y_zoom = QSlider(Qt.Vertical)
        self.slider_y_zoom.setRange(10, 100)
        self.slider_y_zoom.setValue(100)
        self.slider_y_zoom.setInvertedAppearance(True)  # top = more zoomed in
        self.slider_y_zoom.setToolTip("Y-axis zoom: slide up to zoom in")

        self.canvas = MplCanvas()
        self.canvas.set_drag_callback(self._on_fit_window_dragged)

        self.slider_x_zoom = QSlider(Qt.Horizontal)
        self.slider_x_zoom.setRange(10, 100)
        self.slider_x_zoom.setValue(100)
        self.slider_x_zoom.setToolTip("X-axis zoom: slide left to zoom in")

        btn_reset_zoom = QPushButton("↺")
        btn_reset_zoom.setFixedSize(24, 24)
        btn_reset_zoom.setToolTip("Reset zoom to full view")
        btn_reset_zoom.clicked.connect(self._on_reset_zoom)

        canvas_grid.addWidget(self.slider_y_zoom, 0, 0)
        canvas_grid.addWidget(self.canvas, 0, 1)
        canvas_grid.addWidget(btn_reset_zoom, 1, 0)
        canvas_grid.addWidget(self.slider_x_zoom, 1, 1)
        canvas_grid.setColumnStretch(1, 1)
        canvas_grid.setRowStretch(0, 1)

        self.slider_x_zoom.valueChanged.connect(self._on_x_zoom_changed)
        self.slider_y_zoom.valueChanged.connect(self._on_y_zoom_changed)

        right_layout.addWidget(canvas_area, stretch=3)

        self.lbl_output = QPlainTextEdit()
        self.lbl_output.setReadOnly(True)
        self.lbl_output.document().setMaximumBlockCount(2500)
        self.lbl_output.setMinimumHeight(120)
        self.lbl_output.setStyleSheet(
            "QPlainTextEdit { background: #1e1e2e; color: #d4d4d4; padding: 8px; "
            "font-family: Consolas, 'Courier New', monospace; font-size: 12px; }"
        )
        self.lbl_output.setPlainText("Outputs will appear here")
        right_layout.addWidget(self.lbl_output, stretch=1)

        left.setMinimumWidth(400)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        layout = QHBoxLayout(root)
        layout.addWidget(splitter)

        # populate initial model
        self._on_model_changed(self._cmb_model.currentText())
        self._cmb_model.currentTextChanged.connect(self._on_model_changed)

        self._update_view_buttons()

    def _view_colors_path(self) -> Path:
        return Path(__file__).parent / "view_colors.json"

    def _load_view_color_presets(self) -> list[dict]:
        default = [
            {"name": "Blue", "color": "#1f77b4", "role": "sim", "palette": "distinct"},
            {"name": "Orange", "color": "#ff7f0e", "role": "sim", "palette": "distinct"},
            {"name": "Green", "color": "#2ca02c", "role": "exp", "palette": "distinct"},
            {"name": "Red", "color": "#d62728", "role": "exp", "palette": "distinct"},
            {"name": "Purple", "color": "#9467bd", "role": "sim", "palette": "distinct"},
            {"name": "Brown", "color": "#8c564b", "role": "sim", "palette": "distinct"},
            {"name": "Pink", "color": "#e377c2", "role": "exp", "palette": "distinct"},
            {"name": "Gray", "color": "#7f7f7f", "role": "exp", "palette": "distinct"},
            {"name": "Olive", "color": "#bcbd22", "role": "sim", "palette": "distinct"},
            {"name": "Cyan", "color": "#17becf", "role": "sim", "palette": "distinct"},
            {"name": "Black", "color": "#000000", "role": "exp", "palette": "distinct"},
        ]
        path = self._view_colors_path()
        if not path.exists():
            return default
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
        presets: list[dict] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                color = str(item.get("color", "")).strip()
                if not name or not QColor(color).isValid():
                    continue
                role = str(
                    item.get("role", item.get("type", item.get("curve_type", "both")))
                ).strip().lower()
                if role in ("experiment", "experimental"):
                    role = "exp"
                elif role in ("simulation", "fit"):
                    role = "sim"
                if role not in ("exp", "sim", "both", ""):
                    role = "both"
                palette = str(item.get("palette", "distinct")).strip().lower()
                if palette in ("gradient", "sweep_gradient", "sweep-gradient"):
                    palette = "sweep"
                if palette not in ("distinct", "sweep", "both", ""):
                    palette = "distinct"
                presets.append({
                    "name": name,
                    "color": QColor(color).name(),
                    "role": role or "both",
                    "palette": palette or "distinct",
                })
        return presets or default

    def _build_curve_view_dock(self):
        self._curve_view_dock = QDockWidget("Curve View", self)
        self._curve_view_dock.setObjectName("CurveViewDock")
        self._curve_view_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self.chk_multi_curve = QCheckBox("Enable multiple curve selection")
        self.chk_multi_curve.setChecked(False)
        self.chk_multi_curve.toggled.connect(self._on_multi_curve_toggled)
        lay.addWidget(self.chk_multi_curve)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color mode:"))
        self.cmb_curve_color_mode = _NoWheelComboBox()
        self.cmb_curve_color_mode.addItem("Distinct", "distinct")
        self.cmb_curve_color_mode.addItem("Sweep gradient", "sweep")
        self.cmb_curve_color_mode.setToolTip(
            "Distinct uses high-contrast colors. Sweep gradient maps curve order to a parameter-sweep color ramp."
        )
        self.cmb_curve_color_mode.currentIndexChanged.connect(self._on_curve_color_mode_changed)
        color_row.addWidget(self.cmb_curve_color_mode)
        self.btn_apply_curve_colors = QPushButton("Apply colors")
        self.btn_apply_curve_colors.setToolTip(
            "Apply the selected color mode to selected curves, or to all curves if no rows are selected."
        )
        self.btn_apply_curve_colors.clicked.connect(self._on_apply_curve_colors)
        color_row.addWidget(self.btn_apply_curve_colors)
        color_row.addStretch(1)
        lay.addLayout(color_row)

        self._curve_col_widths = {0: 28, 1: 81, 2: 121, 3: 72, 4: 90}
        self._curve_col_min_widths = {0: 28, 1: 28, 2: 64, 3: 28, 4: 35}
        self._curve_row_widgets: list[list[QWidget]] = []

        self._curve_scroll = QScrollArea()
        self._curve_scroll.setWidgetResizable(True)
        self._curve_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._curve_scroll.setMinimumWidth(420)
        self._curve_scroll.viewport().installEventFilter(self)
        self._curve_grid_host = QWidget()
        self._curve_grid = QGridLayout(self._curve_grid_host)
        self._curve_grid.setContentsMargins(0, 0, 0, 0)
        self._curve_grid.setHorizontalSpacing(3)
        self._curve_grid.setVerticalSpacing(4)
        self._curve_grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._curve_grid.setRowStretch(999, 1)
        self._build_curve_grid_header()
        self._curve_scroll.setWidget(self._curve_grid_host)
        lay.addWidget(self._curve_scroll, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_import_curves = QPushButton("Batch import curves")
        self.btn_import_curves.clicked.connect(self._on_batch_import_curves)
        btn_row.addWidget(self.btn_import_curves)
        self.btn_clear_selected_curves = QPushButton("Clear selected")
        self.btn_clear_selected_curves.setEnabled(False)
        self.btn_clear_selected_curves.clicked.connect(self._on_clear_selected_view_curve)
        btn_row.addWidget(self.btn_clear_selected_curves)
        self.btn_clear_curves = QPushButton("Clear all")
        self.btn_clear_curves.setEnabled(False)
        self.btn_clear_curves.clicked.connect(self._on_clear_view_curves)
        btn_row.addWidget(self.btn_clear_curves)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        move_row = QHBoxLayout()
        self.btn_curve_move_top = QPushButton("Move to top")
        self.btn_curve_move_top.setEnabled(False)
        self.btn_curve_move_top.clicked.connect(self._on_move_selected_curves_to_top)
        move_row.addWidget(self.btn_curve_move_top)
        self.btn_curve_move_up = QPushButton("Move up")
        self.btn_curve_move_up.setEnabled(False)
        self.btn_curve_move_up.clicked.connect(self._on_move_selected_curves_up)
        move_row.addWidget(self.btn_curve_move_up)
        self.btn_curve_move_down = QPushButton("Move down")
        self.btn_curve_move_down.setEnabled(False)
        self.btn_curve_move_down.clicked.connect(self._on_move_selected_curves_down)
        move_row.addWidget(self.btn_curve_move_down)
        move_row.addStretch(1)
        lay.addLayout(move_row)

        self._curve_view_dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._curve_view_dock)
        self._curve_view_dock.setFloating(True)
        self._curve_view_dock.resize(560, 420)
        self._curve_view_dock.hide()
        self._curve_view_dock.visibilityChanged.connect(self._on_curve_view_visibility_changed)

    def _toggle_curve_view_panel(self, checked: bool):
        if checked:
            self._curve_view_dock.setFloating(True)
        self._curve_view_dock.setVisible(bool(checked))

    def _on_curve_view_visibility_changed(self, visible: bool):
        if hasattr(self, "_act_curve_view"):
            self._act_curve_view.setChecked(bool(visible))

    def _on_multi_curve_toggled(self, checked: bool):
        self._multi_curve_enabled = bool(checked)
        if checked:
            self._seed_view_curves_from_current_canvas()
        self.statusBar().showMessage(
            "Multiple curve selection enabled" if checked else "Multiple curve selection disabled"
        )
        self._redraw_all()

    def _on_curve_color_mode_changed(self, _index: int):
        if hasattr(self, "cmb_curve_color_mode"):
            self._curve_color_mode = str(self.cmb_curve_color_mode.currentData() or "distinct")

    def _curve_view_active(self) -> bool:
        return (
            bool(getattr(self, "_multi_curve_enabled", False))
            and hasattr(self, "_curve_view_dock")
            and self._curve_view_dock.isVisible()
        )

    def eventFilter(self, obj, event):  # noqa: N802
        if (
            hasattr(self, "_curve_scroll")
            and obj is self._curve_scroll.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            QTimer.singleShot(0, self._fit_curve_columns_to_view)
        if event.type() == QEvent.Type.MouseButtonPress:
            row = obj.property("curve_row") if hasattr(obj, "property") else None
            if row is not None:
                try:
                    col = obj.property("curve_col") if hasattr(obj, "property") else None
                    if int(col) == 0:
                        if isinstance(obj, QCheckBox):
                            obj.setChecked(not obj.isChecked())
                        return True
                    self._select_view_curve_row(int(row), event.modifiers())
                except Exception:
                    pass
        return super().eventFilter(obj, event)

    def _curve_grid_col(self, logical_col: int) -> int:
        mapping = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}
        return mapping.get(logical_col, logical_col * 2)

    def _build_curve_grid_header(self):
        for col in range(12):
            self._curve_grid.setColumnStretch(col, 0)
        for col, text in enumerate(("Show", "Type", "Legend", "Color", "Width")):
            label = QLabel(text)
            label.setStyleSheet("font-weight: 600;")
            label.setFixedWidth(self._curve_col_widths[col])
            label.setFixedHeight(24)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self._curve_grid.addWidget(label, 0, self._curve_grid_col(col))
            if col < 4:
                handle = _ColumnResizeHandle()
                handle.dragged.connect(lambda delta, c=col: self._resize_curve_columns(c, delta))
                self._curve_grid.addWidget(handle, 0, self._curve_grid_col(col) + 1)
        self._curve_grid.setColumnStretch(10, 0)

    def _resize_curve_columns(self, left_col: int, delta: int):
        legend_col = 2
        color_col = 3
        widths = self._curve_col_widths
        mins = self._curve_col_min_widths
        delta = int(delta)

        # Boundary between Legend and Color: keep total width fixed and trade
        # space only between those two columns.
        if left_col == legend_col:
            legend_new = max(mins[legend_col], widths[legend_col] + delta)
            actual_delta = legend_new - widths[legend_col]
            color_new = max(mins[color_col], widths[color_col] - actual_delta)
            actual_delta = widths[color_col] - color_new
            widths[legend_col] = widths[legend_col] + actual_delta
            widths[color_col] = color_new
            self._apply_curve_column_widths()
            return

        # Boundary between Color and Width: keep Color fixed and trade width
        # only between Width and Legend.
        if left_col == color_col:
            width_col = 4
            requested_width = max(mins[width_col], widths[width_col] - delta)
            requested_delta = requested_width - widths[width_col]
            requested_legend = max(mins[legend_col], widths[legend_col] - requested_delta)
            actual_delta = widths[legend_col] - requested_legend
            widths[width_col] = widths[width_col] + actual_delta
            widths[legend_col] = requested_legend
            self._apply_curve_column_widths()
            return

        # Columns to the right of Legend grow into the spacer before Width.
        # Once the fixed columns fill the visible panel, stop instead of
        # pushing Width out and creating horizontal scrolling.
        if left_col > legend_col:
            requested = max(mins[left_col], widths[left_col] + delta)
            if requested > widths[left_col]:
                requested = min(requested, widths[left_col] + self._curve_available_extra_width())
            widths[left_col] = requested
            self._fit_curve_columns_to_view()
            self._apply_curve_column_widths()
            return

        # Columns left of Legend also trade space with Legend.
        left_new = max(mins[left_col], widths[left_col] + delta)
        actual_delta = left_new - widths[left_col]

        legend_new = max(mins[legend_col], widths[legend_col] - actual_delta)
        actual_delta = widths[legend_col] - legend_new
        widths[left_col] = widths[left_col] + actual_delta
        widths[legend_col] = legend_new
        self._apply_curve_column_widths()

    def _fit_curve_columns_to_view(self):
        if not hasattr(self, "_curve_scroll"):
            return
        viewport_width = int(self._curve_scroll.viewport().width())
        if viewport_width <= 0:
            return
        legend_col = 2
        mins = self._curve_col_min_widths
        chrome = 4 * 5 + 10 * 3 + 12
        fixed_width = sum(
            int(width)
            for col, width in self._curve_col_widths.items()
            if col != legend_col
        )
        self._curve_col_widths[legend_col] = max(
            mins[legend_col],
            viewport_width - fixed_width - chrome,
        )
        self._apply_curve_column_widths()

    def _resize_width_left_boundary(self, delta: int):
        legend_col = 2
        width_col = 4
        widths = self._curve_col_widths
        mins = self._curve_col_min_widths
        delta = int(delta)

        # Move Width's left edge while keeping its right edge anchored.
        # Drag right: Width shrinks and Legend grows.
        # Drag left: Width grows and takes space from Legend.
        requested_width = max(mins[width_col], widths[width_col] - delta)
        requested_delta = requested_width - widths[width_col]
        requested_legend = max(mins[legend_col], widths[legend_col] - requested_delta)
        actual_delta = widths[legend_col] - requested_legend
        widths[width_col] = widths[width_col] + actual_delta
        widths[legend_col] = requested_legend
        self._apply_curve_column_widths()

    def _curve_available_extra_width(self) -> int:
        if not hasattr(self, "_curve_scroll"):
            return 0
        viewport_width = int(self._curve_scroll.viewport().width())
        if viewport_width <= 0:
            return 0
        # Four resize handles plus layout gaps and a little safety margin.
        chrome = 4 * 5 + 10 * 3 + 16
        fixed_width = sum(int(v) for v in self._curve_col_widths.values())
        return max(0, viewport_width - fixed_width - chrome)

    def _apply_curve_column_widths(self):
        for row_widgets in getattr(self, "_curve_row_widgets", []):
            for col, widget in enumerate(row_widgets):
                widget.setFixedWidth(self._curve_col_widths[col])
        for col in range(5):
            item = self._curve_grid.itemAtPosition(0, self._curve_grid_col(col))
            if item and item.widget():
                item.widget().setFixedWidth(self._curve_col_widths[col])

    @staticmethod
    def _curve_color_role(curve_type: str) -> str:
        return "exp" if str(curve_type).lower() == "experiment" else "sim"

    def _ordered_view_color_presets(self, curve_type: str, palette: str | None = None) -> list[dict]:
        role = self._curve_color_role(curve_type)
        palette = str(palette or "").lower()
        presets = list(self._view_color_presets or [{"name": "Blue", "color": "#1f77b4", "role": "both"}])
        if palette:
            filtered = [
                p for p in presets
                if str(p.get("palette", "distinct")).lower() in (palette, "both", "")
            ]
            if filtered:
                presets = filtered
        preferred = [p for p in presets if str(p.get("role", "both")).lower() in (role, "both", "")]
        secondary = [p for p in presets if p not in preferred]
        return preferred + secondary

    def _make_curve_color_combo(self, color: str = "#1f77b4", curve_type: str = "simulation") -> QComboBox:
        combo = _NoWheelComboBox()
        combo.setItemDelegate(_ColorSwatchDelegate(combo))
        selected_index = -1
        for i, preset in enumerate(self._ordered_view_color_presets(curve_type)):
            combo.addItem(" ", preset["color"])
            combo.setItemData(i, QColor(preset["color"]), Qt.ItemDataRole.BackgroundRole)
            combo.setItemData(i, QColor("#000000"), Qt.ItemDataRole.ForegroundRole)
            role = str(preset.get("role", "both"))
            palette = str(preset.get("palette", "distinct"))
            combo.setItemData(i, f"{preset['name']} ({role}, {palette})", Qt.ItemDataRole.ToolTipRole)
            if QColor(preset["color"]).name().lower() == QColor(color).name().lower():
                selected_index = i
        combo.addItem("More colors...", "__more__")
        if selected_index < 0 and QColor(color).isValid():
            insert_at = max(0, combo.count() - 1)
            custom = QColor(color).name()
            combo.insertItem(insert_at, custom.upper(), custom)
            combo.setItemData(insert_at, QColor(custom), Qt.ItemDataRole.BackgroundRole)
            combo.setItemData(insert_at, QColor("#000000"), Qt.ItemDataRole.ForegroundRole)
            combo.setItemData(insert_at, "Current custom color", Qt.ItemDataRole.ToolTipRole)
            selected_index = insert_at
        if selected_index >= 0:
            combo.setCurrentIndex(selected_index)
        self._refresh_curve_color_combo(combo)
        return combo

    def _refresh_curve_color_combo(self, combo: QComboBox):
        data = combo.currentData()
        if data == "__more__":
            combo.setStyleSheet("")
            return
        color = QColor(str(data))
        if not color.isValid():
            combo.setStyleSheet("")
            return
        combo.setStyleSheet(
            "QComboBox {"
            f" background-color: {color.name()};"
            " color: transparent;"
            " border: 2px solid #000;"
            " padding: 0 14px 0 2px;"
            " selection-background-color: transparent;"
            "}"
            "QComboBox::drop-down {"
            " border-left: 1px solid #000;"
            " background: rgba(255, 255, 255, 35);"
            " width: 14px;"
            "}"
            "QComboBox QAbstractItemView {"
            " border: 1px solid #000;"
            " outline: 0;"
            " selection-background-color: rgba(255, 255, 255, 45);"
            "}"
        )

    def _next_view_curve_color(self, curve_type: str = "simulation") -> str:
        presets = self._ordered_view_color_presets(curve_type, palette="distinct")
        same_type_count = sum(
            1
            for curve in self._view_curves
            if self._curve_color_role(str(curve.get("type", "simulation")))
            == self._curve_color_role(curve_type)
        )
        idx = same_type_count % len(presets)
        return str(presets[idx].get("color", "#1f77b4"))

    @staticmethod
    def _interpolate_hex_colors(stops: list[str], count: int) -> list[str]:
        valid = [QColor(c) for c in stops if QColor(c).isValid()]
        if count <= 0:
            return []
        if not valid:
            return ["#1f77b4"] * count
        if count == 1:
            return [valid[0].name()]
        if len(valid) == 1:
            return [valid[0].name()] * count
        positions = np.linspace(0.0, len(valid) - 1, count)
        colors: list[str] = []
        for pos in positions:
            lo = int(np.floor(pos))
            hi = min(lo + 1, len(valid) - 1)
            frac = float(pos - lo)
            c0 = valid[lo]
            c1 = valid[hi]
            r = round(c0.red() + (c1.red() - c0.red()) * frac)
            g = round(c0.green() + (c1.green() - c0.green()) * frac)
            b = round(c0.blue() + (c1.blue() - c0.blue()) * frac)
            colors.append(QColor(r, g, b).name())
        return colors

    def _colors_for_curve_targets(self, target_indices: list[int], palette: str) -> list[str]:
        if not target_indices:
            return []
        if palette == "sweep":
            presets = self._ordered_view_color_presets("simulation", palette="sweep")
            stops = [str(p.get("color", "#1f77b4")) for p in presets]
            return self._interpolate_hex_colors(stops, len(target_indices))

        colors: list[str] = []
        counters = {"experiment": 0, "simulation": 0}
        for idx in target_indices:
            curve_type = str(self._view_curves[idx].get("type", "simulation"))
            presets = self._ordered_view_color_presets(curve_type, palette="distinct")
            key = "experiment" if curve_type == "experiment" else "simulation"
            preset = presets[counters[key] % len(presets)]
            colors.append(str(preset.get("color", "#1f77b4")))
            counters[key] += 1
        return colors

    def _on_apply_curve_colors(self):
        if not self._view_curves:
            return
        palette = str(getattr(self, "_curve_color_mode", "distinct") or "distinct")
        if hasattr(self, "cmb_curve_color_mode"):
            palette = str(self.cmb_curve_color_mode.currentData() or palette)
            self._curve_color_mode = palette
        targets = self._selected_curve_index_list()
        if not targets:
            targets = list(range(len(self._view_curves)))
        colors = self._colors_for_curve_targets(targets, palette)
        for idx, color in zip(targets, colors):
            self._view_curves[idx]["color"] = color
        self._rebuild_curve_rows()
        self.statusBar().showMessage(
            f"Applied {'sweep gradient' if palette == 'sweep' else 'distinct'} colors to {len(targets)} curve(s)."
        )

    @staticmethod
    def _normalise_legend_text(text: str) -> str:
        replacements = {
            r"\alpha": "α",
            r"\beta": "β",
            r"\gamma": "γ",
            r"\delta": "δ",
            r"\epsilon": "ε",
            r"\varepsilon": "ε",
            r"\zeta": "ζ",
            r"\eta": "η",
            r"\theta": "θ",
            r"\vartheta": "θ",
            r"\iota": "ι",
            r"\kappa": "κ",
            r"\lambda": "λ",
            r"\mu": "μ",
            r"\nu": "ν",
            r"\xi": "ξ",
            r"\pi": "π",
            r"\rho": "ρ",
            r"\sigma": "σ",
            r"\tau": "τ",
            r"\upsilon": "υ",
            r"\phi": "φ",
            r"\varphi": "φ",
            r"\chi": "χ",
            r"\psi": "ψ",
            r"\omega": "ω",
            r"\Gamma": "Γ",
            r"\Delta": "Δ",
            r"\Theta": "Θ",
            r"\Lambda": "Λ",
            r"\Xi": "Ξ",
            r"\Pi": "Π",
            r"\Sigma": "Σ",
            r"\Phi": "Φ",
            r"\Psi": "Ψ",
            r"\Omega": "Ω",
        }
        out = str(text or "")
        out = out.replace("$", "")
        for src, dst in replacements.items():
            out = out.replace(src, dst)
        return out

    def _curve_legend_from_mat(self, mat: dict, default: str) -> str:
        legend = self._mat_to_string(mat, "legend", "").strip()
        if not legend:
            legend = default
        return self._normalise_legend_text(legend)

    @staticmethod
    def _infer_curve_type(t: np.ndarray, R: np.ndarray) -> str:
        t_arr = np.asarray(t, dtype=float).reshape(-1)
        R_arr = np.asarray(R, dtype=float).reshape(-1)
        n = min(t_arr.size, R_arr.size)
        if n >= 4:
            t_sorted = np.sort(t_arr[:n])
            dt = np.diff(t_sorted)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if dt.size:
                median_dt = float(np.median(dt))
                # Fastest expected experiment is about 10M fps, i.e. 0.1 us.
                # Anything meaningfully denser is treated as simulation/fit.
                if median_dt < 1.0e-7:
                    return "simulation"
            span = float(np.nanmax(t_arr[:n]) - np.nanmin(t_arr[:n]))
            if np.isfinite(span) and span > 0 and n >= 20000:
                return "simulation"
        return "experiment"

    def _seed_view_curves_from_current_canvas(self):
        if self._view_curves:
            return
        if self.state.exp_t is not None and self.state.exp_R is not None:
            name = Path(self.state.exp_path).stem if self.state.exp_path else "exp data"
            self._add_curve_to_view(
                curve_type="experiment",
                t=self.state.exp_t,
                R=self.state.exp_R,
                legend=name,
                redraw=False,
            )
        if self.state.sim_t is not None and self.state.sim_R is not None:
            self._add_curve_to_view(
                curve_type="simulation",
                t=self.state.sim_t,
                R=self.state.sim_R,
                legend=f"{self._get_active_model_key()} simulation",
                meta=self.state.sim_meta,
                redraw=False,
            )

    def _add_curve_to_view(
        self,
        *,
        curve_type: str,
        t: np.ndarray,
        R: np.ndarray,
        legend: str,
        meta: dict | None = None,
        redraw: bool = True,
    ):
        t_arr = np.asarray(t, dtype=float).reshape(-1)
        R_arr = np.asarray(R, dtype=float).reshape(-1)
        n = min(t_arr.size, R_arr.size)
        if n <= 0:
            return
        if curve_type == "auto":
            curve_type = self._infer_curve_type(t_arr[:n], R_arr[:n])
        curve = {
            "visible": True,
            "type": "experiment" if curve_type == "experiment" else "simulation",
            "legend": self._normalise_legend_text(legend),
            "color": self._next_view_curve_color(curve_type),
            "width": 1.5,
            "t": t_arr[:n].copy(),
            "R": R_arr[:n].copy(),
            "meta": dict(meta or {}),
        }
        self._add_view_curve_row(curve)
        if redraw:
            self._redraw_all()

    def _apply_curve_row_selection_style(self):
        for i, row_widgets in enumerate(self._curve_row_widgets):
            style = "background: rgba(80, 140, 220, 50);" if i in self._selected_curve_indices else ""
            for col, widget in enumerate(row_widgets):
                if col == 3:
                    self._refresh_curve_color_combo(widget)
                    continue
                widget.setStyleSheet(style)
        self._update_curve_selection_buttons()

    def _update_curve_selection_buttons(self):
        has_selection = bool(self._selected_curve_indices)
        self.btn_clear_selected_curves.setEnabled(has_selection)
        if hasattr(self, "btn_curve_move_top"):
            self.btn_curve_move_top.setEnabled(has_selection)
            self.btn_curve_move_up.setEnabled(has_selection)
            self.btn_curve_move_down.setEnabled(has_selection)

    def _select_view_curve_row(self, row_index: int, modifiers=Qt.KeyboardModifier.NoModifier):
        if not (0 <= row_index < len(self._view_curves)):
            return
        mods = Qt.KeyboardModifiers(modifiers)
        if mods & Qt.KeyboardModifier.ShiftModifier:
            anchor = self._curve_selection_anchor
            if anchor is None or not (0 <= anchor < len(self._view_curves)):
                anchor = row_index
            start, end = sorted((anchor, row_index))
            if mods & Qt.KeyboardModifier.ControlModifier:
                self._selected_curve_indices.update(range(start, end + 1))
            else:
                self._selected_curve_indices = set(range(start, end + 1))
        elif mods & Qt.KeyboardModifier.ControlModifier:
            if row_index in self._selected_curve_indices:
                self._selected_curve_indices.remove(row_index)
            else:
                self._selected_curve_indices.add(row_index)
            self._curve_selection_anchor = row_index
        else:
            self._selected_curve_indices = {row_index}
            self._curve_selection_anchor = row_index
        self._apply_curve_row_selection_style()

    def _add_view_curve_row(self, curve: dict):
        row_index = len(self._view_curves)
        row = row_index + 1
        self._view_curves.append(curve)

        chk_show = QCheckBox()
        chk_show.setChecked(bool(curve.get("visible", True)))
        chk_show.setToolTip("Show or hide this curve in the preview.")
        chk_show.setFixedWidth(self._curve_col_widths[0])
        self._curve_grid.addWidget(chk_show, row, self._curve_grid_col(0))

        cmb_type = _NoWheelComboBox()
        cmb_type.addItem("······", "experiment")
        cmb_type.addItem("━━━━", "simulation")
        curve_type = str(curve.get("type", "simulation")).lower()
        cmb_type.setCurrentIndex(0 if curve_type == "experiment" else 1)
        cmb_type.setToolTip("Curve style. Dots use markers; line uses a solid curve.")
        cmb_type.setEditable(False)
        cmb_type.setFixedWidth(self._curve_col_widths[1])
        self._curve_grid.addWidget(cmb_type, row, self._curve_grid_col(1))

        le_legend = QLineEdit(str(curve.get("legend", f"curve {row}")))
        le_legend.setToolTip("Legend label shown in the preview.")
        le_legend.setFixedWidth(self._curve_col_widths[2])
        self._curve_grid.addWidget(le_legend, row, self._curve_grid_col(2))

        cmb_color = self._make_curve_color_combo(
            str(curve.get("color", "#1f77b4")),
            str(curve.get("type", "simulation")),
        )
        cmb_color.setToolTip("Curve color. Choose More colors... for a custom color.")
        cmb_color.setFixedWidth(self._curve_col_widths[3])
        self._curve_grid.addWidget(cmb_color, row, self._curve_grid_col(3))

        spin_width = _NoWheelSpinBox()
        spin_width.setRange(0.25, 10.0)
        spin_width.setSingleStep(0.25)
        spin_width.setDecimals(2)
        spin_width.setValue(float(curve.get("width", 1.5)))
        spin_width.setFixedWidth(self._curve_col_widths[4])
        spin_width.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._curve_grid.addWidget(spin_width, row, self._curve_grid_col(4))
        self._curve_row_widgets.append([chk_show, cmb_type, le_legend, cmb_color, spin_width])
        for col_index, widget in enumerate(self._curve_row_widgets[-1]):
            widget.installEventFilter(self)
            widget.setProperty("curve_row", row_index)
            widget.setProperty("curve_col", col_index)

        def _sync_curve():
            curve["visible"] = chk_show.isChecked()
            curve["type"] = cmb_type.currentData()
            legend_text = self._normalise_legend_text(le_legend.text().strip() or f"curve {row}")
            curve["legend"] = legend_text
            if le_legend.text() != legend_text:
                le_legend.blockSignals(True)
                le_legend.setText(legend_text)
                le_legend.blockSignals(False)
            color_data = cmb_color.currentData()
            if color_data != "__more__":
                curve["color"] = str(color_data)
            curve["width"] = float(spin_width.value())
            self._redraw_all()

        def _choose_color(index: int):
            item_data = cmb_color.itemData(index)
            if item_data != "__more__":
                _sync_curve()
                return
            initial = QColor(str(curve.get("color", "#1f77b4")))
            chosen = QColorDialog.getColor(initial, self, "Select curve color")
            if chosen.isValid():
                label = chosen.name().upper()
                insert_at = max(0, cmb_color.count() - 1)
                cmb_color.insertItem(insert_at, label, chosen.name())
                cmb_color.setCurrentIndex(insert_at)
                curve["color"] = chosen.name()
            else:
                current = str(curve.get("color", "#1f77b4"))
                def _restore_current_color():
                    restored = False
                    for i in range(cmb_color.count()):
                        data = cmb_color.itemData(i)
                        if data != "__more__" and str(data).lower() == QColor(current).name().lower():
                            cmb_color.blockSignals(True)
                            cmb_color.setCurrentIndex(i)
                            cmb_color.blockSignals(False)
                            restored = True
                            break
                    if not restored:
                        insert_at = max(0, cmb_color.count() - 1)
                        cmb_color.insertItem(insert_at, QColor(current).name().upper(), QColor(current).name())
                        cmb_color.blockSignals(True)
                        cmb_color.setCurrentIndex(insert_at)
                        cmb_color.blockSignals(False)
                    self._refresh_curve_color_combo(cmb_color)
                    _sync_curve()
                QTimer.singleShot(0, _restore_current_color)
                return
            self._refresh_curve_color_combo(cmb_color)
            _sync_curve()

        chk_show.toggled.connect(lambda _checked: _sync_curve())
        cmb_type.currentIndexChanged.connect(lambda _idx: _sync_curve())
        le_legend.editingFinished.connect(_sync_curve)
        cmb_color.currentIndexChanged.connect(lambda idx: (self._refresh_curve_color_combo(cmb_color), _choose_color(idx)))
        spin_width.valueChanged.connect(lambda _value: _sync_curve())
        self.btn_clear_curves.setEnabled(True)
        self._update_curve_selection_buttons()
        _sync_curve()

    def _on_clear_view_curves(self):
        if not self._view_curves:
            return
        reply = QMessageBox.question(
            self,
            "Clear curves",
            "Clear all curves from the Curve View panel?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._view_curves.clear()
        self._selected_curve_indices.clear()
        self._curve_selection_anchor = None
        while self._curve_grid.count() > 0:
            item = self._curve_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._curve_row_widgets.clear()
        self._curve_grid.setRowStretch(999, 1)
        self._build_curve_grid_header()
        self.btn_clear_curves.setEnabled(False)
        self._update_curve_selection_buttons()
        self._redraw_all()

    def _rebuild_curve_rows(self):
        curves = list(self._view_curves)
        while self._curve_grid.count() > 0:
            item = self._curve_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._view_curves = []
        self._curve_row_widgets.clear()
        self._build_curve_grid_header()
        for curve in curves:
            self._add_view_curve_row(curve)
        self._selected_curve_indices = {
            i for i in self._selected_curve_indices if i < len(self._view_curves)
        }
        self._curve_selection_anchor = None
        self._apply_curve_row_selection_style()
        self.btn_clear_curves.setEnabled(bool(self._view_curves))
        self._update_curve_selection_buttons()
        self._redraw_all()

    def _on_clear_selected_view_curve(self):
        remove_indices = {
            i for i in self._selected_curve_indices if 0 <= i < len(self._view_curves)
        }
        if not remove_indices:
            return
        self._view_curves = [
            curve
            for i, curve in enumerate(self._view_curves)
            if i not in remove_indices
        ]
        self._selected_curve_indices.clear()
        self._curve_selection_anchor = None
        self._rebuild_curve_rows()

    def _selected_curve_index_list(self) -> list[int]:
        return sorted(
            i for i in self._selected_curve_indices if 0 <= i < len(self._view_curves)
        )

    def _set_curve_order_and_selection(self, curves: list[dict], selected_old_indices: set[int]):
        selected_ids = {id(self._view_curves[i]) for i in selected_old_indices}
        self._view_curves = curves
        self._selected_curve_indices = {
            i for i, curve in enumerate(self._view_curves) if id(curve) in selected_ids
        }
        self._curve_selection_anchor = min(self._selected_curve_indices) if self._selected_curve_indices else None
        self._rebuild_curve_rows()

    def _on_move_selected_curves_to_top(self):
        selected = set(self._selected_curve_index_list())
        if not selected:
            return
        moved = [curve for i, curve in enumerate(self._view_curves) if i in selected]
        rest = [curve for i, curve in enumerate(self._view_curves) if i not in selected]
        self._set_curve_order_and_selection(moved + rest, selected)

    def _on_move_selected_curves_up(self):
        selected = set(self._selected_curve_index_list())
        if not selected:
            return
        curves = list(self._view_curves)
        for i in range(1, len(curves)):
            if i in selected and (i - 1) not in selected:
                curves[i - 1], curves[i] = curves[i], curves[i - 1]
        self._set_curve_order_and_selection(curves, selected)

    def _on_move_selected_curves_down(self):
        selected = set(self._selected_curve_index_list())
        if not selected:
            return
        curves = list(self._view_curves)
        for i in range(len(curves) - 2, -1, -1):
            if i in selected and (i + 1) not in selected:
                curves[i + 1], curves[i] = curves[i], curves[i + 1]
        self._set_curve_order_and_selection(curves, selected)

    def _on_batch_import_curves(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Batch import curves",
            "",
            "MAT files (*.mat)",
        )
        if not paths:
            return
        if not self._curve_view_dock.isVisible():
            self._curve_view_dock.setFloating(True)
            self._curve_view_dock.show()
        if not self.chk_multi_curve.isChecked():
            self.chk_multi_curve.setChecked(True)
        added = 0
        skipped: list[str] = []
        for path in paths:
            try:
                mat = loadmat(path, squeeze_me=True, struct_as_record=False)
                base_legend = self._curve_legend_from_mat(mat, Path(path).stem)
                t_sim = self._mat_array(mat, "t_sim")
                R_sim = self._mat_array(mat, "R_sim")
                t_exp = self._mat_array(mat, "t_exp")
                R_exp = self._mat_array(mat, "R_exp")
                has_exp_curve = t_exp.size >= 3 and R_exp.size >= 3
                has_sim_curve = t_sim.size >= 3 and R_sim.size >= 3
                if t_exp.size >= 3 and R_exp.size >= 3:
                    self._add_curve_to_view(
                        curve_type="auto",
                        t=t_exp,
                        R=R_exp,
                        legend=f"{base_legend} exp" if has_sim_curve else base_legend,
                        redraw=False,
                    )
                    added += 1
                if t_sim.size >= 3 and R_sim.size >= 3:
                    meta = {
                        "Rmax": self._mat_to_float(mat, "Rmax_sim", float(np.nanmax(R_sim))),
                        "t_rmax": 0.0,
                        "tc": self._mat_to_float(mat, "tc", 1.0),
                    }
                    self._add_curve_to_view(
                        curve_type="simulation",
                        t=t_sim,
                        R=R_sim,
                        legend=f"{base_legend} sim" if has_exp_curve else base_legend,
                        meta=meta,
                        redraw=False,
                    )
                    added += 1
                if t_exp.size >= 3 or t_sim.size >= 3:
                    continue
                exp = load_experiment_mat(path)
                self._add_curve_to_view(
                    curve_type="auto",
                    t=exp.t,
                    R=exp.R,
                    legend=base_legend,
                    redraw=False,
                )
                added += 1
            except Exception as exc:
                skipped.append(f"{Path(path).name}: {exc}")
        self._redraw_all()
        self.statusBar().showMessage(f"Imported {added} curve(s).")
        self.lbl_output.appendPlainText(f"Imported {added} curve(s) into Curve View.")
        if skipped:
            detail = "\n".join(skipped[:20])
            if len(skipped) > 20:
                detail += f"\n... and {len(skipped) - 20} more"
            QMessageBox.warning(
                self,
                "Some curves were skipped",
                f"Imported {added} curve(s), skipped {len(skipped)} file(s).\n\n{detail}",
            )

    def _view_export_curves(self) -> list[dict]:
        return [
            curve
            for curve in self._view_curves
            if curve.get("visible", True)
        ] or list(self._view_curves)

    def _view_export_stem(self) -> str:
        if self.state.exp_path:
            return f"{Path(self.state.exp_path).stem}_view"
        return f"imr_view_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _on_export_view_mat(self):
        curves = self._view_export_curves()
        if not curves:
            QMessageBox.information(self, "No curves", "There are no curves to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Curve View as .mat",
            f"{self._view_export_stem()}.mat",
            "MAT files (*.mat)",
        )
        if not path:
            return

        n = len(curves)
        t_cells = np.empty((1, n), dtype=object)
        r_cells = np.empty((1, n), dtype=object)
        legends = np.empty((1, n), dtype=object)
        types = np.empty((1, n), dtype=object)
        colors = np.empty((1, n), dtype=object)
        widths = np.zeros((1, n), dtype=float)
        visible = np.zeros((1, n), dtype=bool)
        for i, curve in enumerate(curves):
            t_cells[0, i] = np.asarray(curve.get("t", []), dtype=float).reshape(-1)
            r_cells[0, i] = np.asarray(curve.get("R", []), dtype=float).reshape(-1)
            legends[0, i] = str(curve.get("legend", f"curve {i + 1}"))
            types[0, i] = str(curve.get("type", "simulation"))
            colors[0, i] = str(curve.get("color", "#1f77b4"))
            widths[0, i] = float(curve.get("width", 1.5))
            visible[0, i] = bool(curve.get("visible", True))

        savemat(
            path,
            {
                "imr_view_format": "IMR fitting GUI curve view",
                "view_mode": self.state.view_mode,
                "time_unit": "s",
                "radius_unit": "m",
                "curve_t": t_cells,
                "curve_R": r_cells,
                "curve_legend": legends,
                "curve_type": types,
                "curve_color": colors,
                "curve_width": widths,
                "curve_visible": visible,
                "n_curves": n,
            },
            do_compression=False,
        )
        self.statusBar().showMessage(f"Exported Curve View MAT: {path}")

    def _with_transparent_canvas(self, callback):
        fig = self.canvas.figure
        ax = self.canvas.ax
        old_fig_alpha = fig.patch.get_alpha()
        old_ax_alpha = ax.patch.get_alpha()
        old_fig_face = fig.patch.get_facecolor()
        old_ax_face = ax.patch.get_facecolor()
        try:
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            return callback()
        finally:
            fig.patch.set_facecolor(old_fig_face)
            ax.patch.set_facecolor(old_ax_face)
            fig.patch.set_alpha(old_fig_alpha)
            ax.patch.set_alpha(old_ax_alpha)
            self.canvas.draw_idle()

    def _on_export_view_svg(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Curve View as .svg",
            f"{self._view_export_stem()}.svg",
            "SVG files (*.svg)",
        )
        if not path:
            return
        self._redraw_all()

        def _save():
            self.canvas.figure.savefig(
                path,
                format="svg",
                transparent=True,
                facecolor="none",
                edgecolor="none",
                bbox_inches="tight",
            )

        self._with_transparent_canvas(_save)
        self.statusBar().showMessage(f"Exported Curve View SVG: {path}")

    def _on_copy_view_png(self):
        self._redraw_all()

        def _copy():
            buf = BytesIO()
            self.canvas.figure.savefig(
                buf,
                format="png",
                dpi=300,
                transparent=True,
                facecolor="none",
                edgecolor="none",
                bbox_inches="tight",
            )
            png_data = buf.getvalue()
            image = QImage.fromData(png_data)
            if image.isNull():
                raise RuntimeError("Failed to render Curve View image.")
            mime = QMimeData()
            mime.setData("image/png", QByteArray(png_data))
            mime.setImageData(image)
            QApplication.clipboard().setMimeData(mime)

        try:
            self._with_transparent_canvas(_copy)
        except Exception as exc:
            QMessageBox.warning(self, "Copy view failed", str(exc))
            return
        self.statusBar().showMessage("Copied Curve View PNG to clipboard.")

    def _on_copy_view_svg(self):
        self._redraw_all()

        def _copy():
            buf = BytesIO()
            self.canvas.figure.savefig(
                buf,
                format="svg",
                transparent=True,
                facecolor="none",
                edgecolor="none",
                bbox_inches="tight",
            )
            svg_data = buf.getvalue()
            if not svg_data:
                raise RuntimeError("Failed to render Curve View SVG.")
            mime = QMimeData()
            mime.setData("image/svg+xml", QByteArray(svg_data))
            try:
                mime.setText(svg_data.decode("utf-8", errors="replace"))
            except Exception:
                pass
            QApplication.clipboard().setMimeData(mime)

        try:
            self._with_transparent_canvas(_copy)
        except Exception as exc:
            QMessageBox.warning(self, "Copy view failed", str(exc))
            return
        self.statusBar().showMessage("Copied Curve View SVG to clipboard.")

    # =====================================================================
    # =====================================================================
    # Physics Settings dialog
    # =====================================================================

    def _build_physics_settings(self):
        """Create the persistent widgets used in the Physics Settings dialog."""
        self.spin_P_inf = _NoWheelSpinBox()
        self.spin_P_inf.setRange(1.0, 1e9)
        self.spin_P_inf.setDecimals(1)
        self.spin_P_inf.setValue(101325.0)
        self.spin_P_inf.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.spin_rho = _NoWheelSpinBox()
        self.spin_rho.setRange(1.0, 1e6)
        self.spin_rho.setDecimals(1)
        self.spin_rho.setValue(998.0)
        self.spin_rho.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.spin_c_long = _NoWheelSpinBox()
        self.spin_c_long.setRange(1.0, 1e5)
        self.spin_c_long.setDecimals(1)
        self.spin_c_long.setValue(1485.0)
        self.spin_c_long.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin_c_long.setToolTip("Longitudinal speed of sound in liquid (m/s)")

        self.spin_gamma = _NoWheelSpinBox()
        self.spin_gamma.setRange(0.0, 10.0)
        self.spin_gamma.setDecimals(6)
        self.spin_gamma.setSingleStep(0.001)
        self.spin_gamma.setValue(0.056)
        self.spin_gamma.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin_gamma.setToolTip("Liquid surface tension (N/m)")

        self.spin_NT = _NoWheelSpinBox()
        self.spin_NT.setRange(50, 2000)
        self.spin_NT.setDecimals(0)
        self.spin_NT.setSingleStep(50)
        self.spin_NT.setValue(500)
        self.spin_NT.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self._cmb_solver = _NoWheelComboBox()
        self._cmb_solver.addItems(["BDF", "Radau", "LSODA"])
        self._cmb_solver.setCurrentText("BDF")
        self._cmb_solver.setToolTip(
            "Radau: L-stable implicit RK (closest to MATLAB ode23tb)\n"
            "BDF: variable-order backward-differentiation (fast, stiff)\n"
            "LSODA: auto-switching Adams/BDF"
        )

        self.le_rtol = QLineEdit("1e-8")
        self.le_rtol.setToolTip("Relative tolerance for ODE solver (e.g. 1e-8)")
        self.le_rtol.setPlaceholderText("e.g. 1e-8")

        self.le_atol = QLineEdit("1e-7")
        self.le_atol.setToolTip("Absolute tolerance for ODE solver (e.g. 1e-7)")
        self.le_atol.setPlaceholderText("e.g. 1e-7")

    def _show_physics_settings(self):
        """Open a modal dialog for P_inf, rho, NT, and solver settings."""
        if not hasattr(self, "_physics_dlg"):
            dlg = QDialog(self)
            dlg.setWindowTitle("Physics Settings")
            dlg.setMinimumWidth(360)
            lay = QVBoxLayout(dlg)
            lay.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

            warn_lbl = QLabel(
                "⚠  Warning: do not modify unless you know what you are doing!"
            )
            warn_lbl.setStyleSheet(
                "QLabel { color: #cc6600; font-weight: bold; padding: 4px; "
                "border: 1px solid #cc6600; border-radius: 3px; }"
            )
            warn_lbl.setWordWrap(True)
            warn_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            lay.addWidget(warn_lbl)

            form = QFormLayout()
            form.addRow("P_inf (Pa):", self.spin_P_inf)
            form.addRow("rho (kg/m³):", self.spin_rho)
            form.addRow("c_long (m/s):", self.spin_c_long)
            form.addRow("gamma (N/m):", self.spin_gamma)
            form.addRow("NT (grid):", self.spin_NT)
            lay.addLayout(form)

            # -- collapsible Advanced Solver Settings --
            self._adv_toggle = QToolButton()
            self._adv_toggle.setText("▸ Advanced Solver Settings")
            self._adv_toggle.setCheckable(True)
            self._adv_toggle.setChecked(False)
            self._adv_toggle.setStyleSheet("QToolButton { border: none; }")
            self._adv_toggle.setToolButtonStyle(Qt.ToolButtonTextOnly)
            lay.addWidget(self._adv_toggle)

            self._adv_solver_widget = QWidget()
            adv_form = QFormLayout(self._adv_solver_widget)
            adv_form.setContentsMargins(12, 0, 0, 0)
            adv_form.addRow("ODE solver:", self._cmb_solver)
            adv_form.addRow("RelTol:", self.le_rtol)
            adv_form.addRow("AbsTol:", self.le_atol)
            self._adv_solver_widget.setVisible(False)
            lay.addWidget(self._adv_solver_widget)

            self._adv_toggle.toggled.connect(self._toggle_advanced_solver)

            btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            btn_box.rejected.connect(lambda: (self._save_settings(), dlg.accept()))
            lay.addWidget(btn_box)

            self._physics_dlg = dlg

        self._toggle_advanced_solver(self._adv_toggle.isChecked())
        self._physics_dlg.exec()

    def _toggle_advanced_solver(self, checked: bool):
        self._adv_solver_widget.setVisible(checked)
        self._adv_toggle.setText(
            "▾ Advanced Solver Settings" if checked
            else "▸ Advanced Solver Settings"
        )
        if hasattr(self, "_physics_dlg"):
            self._physics_dlg.layout().activate()
            self._physics_dlg.adjustSize()

    # =====================================================================
    # Optimizer Settings dialog
    # =====================================================================

    def _show_optimizer_settings(self):
        """Open the Optimizer Settings dialog (built once, reused)."""
        if not hasattr(self, "_opt_dlg"):
            self._build_optimizer_settings_dlg()
        self._opt_dlg_load()   # sync widgets → self._opt_config
        self._opt_dlg.exec()

    def _show_job_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Job List Settings")
        dlg.setMinimumSize(760, 220)
        lay = QVBoxLayout(dlg)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        chk_threshold = QCheckBox("Mark job failed when LSQErr exceeds")
        chk_threshold.setChecked(self._job_success_threshold_enabled)
        spin_threshold = _SigFigSpinBox()
        spin_threshold.setRange(0.0, 1e15)
        spin_threshold.setDecimals(6)
        spin_threshold.setValue(float(self._job_success_lsqerr))
        spin_threshold.setToolTip(
            "Final queue jobs with LSQErr above this value are marked failed. "
            "This is separate from optimizer f_tol."
        )
        spin_threshold.setEnabled(chk_threshold.isChecked())
        chk_threshold.toggled.connect(spin_threshold.setEnabled)

        output_row = QHBoxLayout()
        le_output_dir = QLineEdit(self._job_output_dir)
        le_output_dir.setPlaceholderText("Choose output folder...")
        btn_browse = QPushButton("Browse...")
        output_row.addWidget(le_output_dir, stretch=1)
        output_row.addWidget(btn_browse)

        chk_ask_dir = QCheckBox("Ask before running queue")
        chk_ask_dir.setChecked(self._job_ask_output_dir)
        chk_chain_initial = QCheckBox("Use previous job best fit as next initial")
        chk_chain_initial.setChecked(self._job_chain_best_fit_initial)
        chk_chain_initial.setToolTip(
            "When running a queue, seed each queued job with the previous completed job's "
            "best-fit parameters if both jobs use the same model."
        )
        spin_parallel = QSpinBox()
        spin_parallel.setRange(1, 64)
        spin_parallel.setValue(int(self._job_parallel_workers))
        spin_parallel.setToolTip(
            "Maximum number of queued fitting jobs to run at the same time. "
            "If previous-best-fit chaining is enabled, the queue runs serially."
        )
        le_cleanup_regex = QLineEdit(self._job_import_name_cleanup_regex)
        le_cleanup_regex.setPlaceholderText(r"e.g. ^job_\d+_?; _SNOD; _result; _converted")
        le_cleanup_regex.setToolTip(
            "Semicolon-separated regex patterns removed from experiment names "
            "when importing jobs/results. Leave empty to keep names unchanged."
        )
        chk_cleanup_regex = QCheckBox("Enable")
        chk_cleanup_regex.setChecked(self._job_import_name_cleanup_enabled)
        le_cleanup_regex.setEnabled(chk_cleanup_regex.isChecked())
        chk_cleanup_regex.toggled.connect(le_cleanup_regex.setEnabled)
        btn_cleanup_info = QToolButton()
        btn_cleanup_info.setText("?")
        btn_cleanup_info.setAutoRaise(True)
        btn_cleanup_info.setToolTip(
            "Optional semicolon-separated Python regular expressions used only "
            "when importing jobs/results.\n"
            "Each pattern is removed in order from the displayed experiment name.\n\n"
            "Example:\n"
            r"^job_\d+_?; _SNOD; _result; _converted"
            "\ncan turn\n"
            "job_001_Jin_13_42_19_converted_SNOD_result.mat\n"
            "into\n"
            "Jin_13_42_19.mat"
        )
        cleanup_row = QHBoxLayout()
        cleanup_row.addWidget(le_cleanup_regex, stretch=1)
        cleanup_row.addWidget(btn_cleanup_info)

        def _browse_output_dir():
            start_dir = le_output_dir.text().strip()
            if not start_dir or not Path(start_dir).exists():
                start_dir = ""
            path = QFileDialog.getExistingDirectory(self, "Select default job output folder", start_dir)
            if path:
                le_output_dir.setText(path)

        btn_browse.clicked.connect(_browse_output_dir)

        grid.addWidget(QLabel("Parallel workers:"), 0, 1)
        grid.addWidget(spin_parallel, 0, 2)
        grid.addWidget(chk_threshold, 1, 0)
        grid.addWidget(QLabel("Success LSQErr threshold:"), 1, 1)
        grid.addWidget(spin_threshold, 1, 2)
        grid.addWidget(chk_ask_dir, 2, 0)
        grid.addWidget(QLabel("Default output folder:"), 2, 1)
        grid.addLayout(output_row, 2, 2)
        grid.addWidget(chk_cleanup_regex, 3, 0)
        grid.addWidget(QLabel("Import name cleanup regex:"), 3, 1)
        grid.addLayout(cleanup_row, 3, 2)
        grid.addWidget(chk_chain_initial, 4, 0, 1, 3)
        grid.setColumnStretch(2, 1)
        lay.addLayout(grid)
        lay.addStretch(1)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )

        def _save():
            self._job_success_threshold_enabled = chk_threshold.isChecked()
            self._job_success_lsqerr = float(spin_threshold.value())
            self._job_output_dir = le_output_dir.text().strip()
            self._job_ask_output_dir = chk_ask_dir.isChecked()
            self._job_chain_best_fit_initial = chk_chain_initial.isChecked()
            self._job_parallel_workers = int(spin_parallel.value())
            cleanup_regex = le_cleanup_regex.text().strip()
            if chk_cleanup_regex.isChecked() and cleanup_regex:
                for pattern in self._split_cleanup_patterns(cleanup_regex):
                    try:
                        re.compile(pattern)
                    except re.error as exc:
                        QMessageBox.warning(
                            dlg,
                            "Invalid regex",
                            f"Import name cleanup regex is invalid:\n\n"
                            f"{pattern}\n\n{exc}",
                        )
                        return
            self._job_import_name_cleanup_enabled = chk_cleanup_regex.isChecked()
            self._job_import_name_cleanup_regex = cleanup_regex
            self._save_settings()
            dlg.accept()

        btn_box.accepted.connect(_save)
        btn_box.rejected.connect(dlg.reject)
        lay.addWidget(btn_box)
        dlg.exec()

    def _build_optimizer_settings_dlg(self):
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QFormLayout,
            QSpinBox, QCheckBox,
            QGroupBox, QStackedWidget, QLabel, QDialogButtonBox,
            QWidget,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("Optimizer Settings")
        dlg.setMinimumWidth(420)
        root = QVBoxLayout(dlg)

        # ── Algorithm ──────────────────────────────────────────────────
        form_top = QFormLayout()
        self._cmb_opt_method = _NoWheelComboBox()
        methods = [m for m in AVAILABLE_METHODS
                   if m != "CMA-ES" or _HAS_CMA]
        self._cmb_opt_method.addItems(methods)
        form_top.addRow("Algorithm:", self._cmb_opt_method)
        root.addLayout(form_top)

        # ── Shared options ─────────────────────────────────────────────
        grp_shared = QGroupBox("General")
        fs = QFormLayout(grp_shared)

        self._spin_opt_workers = QSpinBox()
        self._spin_opt_workers.setRange(1, 64)
        self._spin_opt_workers.setToolTip(
            "Parallel workers for Differential Evolution and Pattern Search.\n"
            "Uses multiprocessing (separate processes, no GIL limit).\n"
            "Set >1 only for these two algorithms."
        )
        fs.addRow("Workers (DE / PS):", self._spin_opt_workers)

        self._spin_opt_maxfev = QSpinBox()
        self._spin_opt_maxfev.setRange(10, 100_000)
        self._spin_opt_maxfev.setSingleStep(100)
        fs.addRow("Max evaluations:", self._spin_opt_maxfev)

        self._spin_opt_xtol = _SigFigSpinBox()
        self._spin_opt_xtol.setRange(1e-12, 1.0)
        self._spin_opt_xtol.setDecimals(6)
        fs.addRow("Parameter tol (xatol):", self._spin_opt_xtol)

        self._spin_opt_ftol = _SigFigSpinBox()
        self._spin_opt_ftol.setRange(1e-12, 1e6)
        self._spin_opt_ftol.setDecimals(6)
        fs.addRow("Function tol (fatol):", self._spin_opt_ftol)

        root.addWidget(grp_shared)

        # ── Per-algorithm stacked panel ────────────────────────────────
        self._opt_stack = QStackedWidget()

        # page 0 – Nelder-Mead
        pg_nm = QWidget()
        fnm = QFormLayout(pg_nm)
        self._chk_nm_adaptive = QCheckBox("Adaptive simplex")
        self._chk_nm_adaptive.setToolTip(
            "Scales simplex parameters to the number of dimensions.\n"
            "Recommended for >2 fitting parameters."
        )
        fnm.addRow("", self._chk_nm_adaptive)
        self._opt_stack.addWidget(pg_nm)        # index 0

        # page 1 – Powell
        pg_pw = QWidget()
        fpw = QFormLayout(pg_pw)
        lbl_pw_warn = QLabel(
            "⚠ Powell performs one line-search per parameter per iteration.\n"
            "With expensive ODE solves this can take several minutes per\n"
            "iteration. Consider Nelder-Mead or DE for IMR fitting."
        )
        lbl_pw_warn.setWordWrap(True)
        fpw.addRow(lbl_pw_warn)
        self._opt_stack.addWidget(pg_pw)        # index 1

        # page 2 – Pattern Search
        pg_ps = QWidget()
        fps = QFormLayout(pg_ps)
        self._chk_ps_complete = QCheckBox("Complete polling")
        self._chk_ps_complete.setToolTip(
            "If checked: evaluate all 2N directions before accepting.\n"
            "If unchecked (default): accept first improvement found (faster).\n"
            "Matches MATLAB patternsearch CompletePoll='off' default."
        )
        fps.addRow("", self._chk_ps_complete)
        self._spin_ps_mesh_init = _SigFigSpinBox()
        self._spin_ps_mesh_init.setRange(1e-4, 10.0)
        self._spin_ps_mesh_init.setDecimals(4)
        self._spin_ps_mesh_init.setToolTip(
            "Initial poll step size in optimizer space.\n"
            "For log-scaled params: step = this value in log10 units\n"
            "  (0.3 ~= factor-of-2 step, comparable to MATLAB default).\n"
            "For lin-scaled params: step = this value * bounds range."
        )
        fps.addRow("Initial mesh size:", self._spin_ps_mesh_init)
        self._spin_ps_expand = _SigFigSpinBox()
        self._spin_ps_expand.setRange(1.01, 10.0)
        self._spin_ps_expand.setDecimals(4)
        self._spin_ps_expand.setToolTip("Mesh expansion factor on success (MATLAB default: 2.0).")
        fps.addRow("Mesh expansion:", self._spin_ps_expand)
        self._spin_ps_contract = _SigFigSpinBox()
        self._spin_ps_contract.setRange(0.01, 0.99)
        self._spin_ps_contract.setDecimals(4)
        self._spin_ps_contract.setToolTip("Mesh contraction factor on failure (MATLAB default: 0.5).")
        fps.addRow("Mesh contraction:", self._spin_ps_contract)
        self._spin_ps_search = QSpinBox()
        self._spin_ps_search.setRange(0, 10000)
        self._spin_ps_search.setToolTip(
            "Random points sampled before each poll step (search step).\n"
            "0 = no search, pure GPS polling."
        )
        fps.addRow("Search points:", self._spin_ps_search)
        self._chk_ps_debug = QCheckBox("Write debug CSV logs")
        self._chk_ps_debug.setToolTip(
            "Log every function evaluation and every iteration to two CSV files\n"
            "(ps_debug_eval_<stamp>.csv and ps_debug_iter_<stamp>.csv)\n"
            "in the working directory. Use for step-by-step comparison with MATLAB.\n"
            "Only works in sequential mode (workers = 1)."
        )
        fps.addRow("Debug logging:", self._chk_ps_debug)
        self._opt_stack.addWidget(pg_ps)        # index 2

        # page 3 – Newton-CG
        pg_ncg = QWidget()
        fncg = QFormLayout(pg_ncg)
        lbl_ncg = QLabel(
            "Uses finite-difference gradients in optimizer space and a bounded "
            "sigmoid transform. Best used as a local polish step from a good "
            "initial guess; each iteration can require several ODE solves."
        )
        lbl_ncg.setWordWrap(True)
        fncg.addRow(lbl_ncg)
        self._chk_ncg_debug = QCheckBox("Write debug CSV logs")
        self._chk_ncg_debug.setToolTip(
            "Log every Newton-CG objective evaluation and callback iteration\n"
            "to CSV files (ncg_debug_eval_<stamp>.csv and\n"
            "ncg_debug_iter_<stamp>.csv) in the working directory."
        )
        fncg.addRow("Debug logging:", self._chk_ncg_debug)
        self._opt_stack.addWidget(pg_ncg)       # index 3

        # page 4 – Differential Evolution
        pg_de = QWidget()
        fde = QFormLayout(pg_de)
        self._cmb_de_strategy = _NoWheelComboBox()
        self._cmb_de_strategy.addItems(DE_STRATEGIES)
        fde.addRow("Strategy:", self._cmb_de_strategy)
        self._spin_de_maxiter = QSpinBox()
        self._spin_de_maxiter.setRange(1, 10_000)
        fde.addRow("Max iterations:", self._spin_de_maxiter)
        self._spin_de_popsize = QSpinBox()
        self._spin_de_popsize.setRange(2, 200)
        fde.addRow("Population size:", self._spin_de_popsize)
        self._spin_de_mutation = _SigFigSpinBox()
        self._spin_de_mutation.setRange(0.0, 2.0)
        self._spin_de_mutation.setDecimals(3)
        fde.addRow("Mutation (F):", self._spin_de_mutation)
        self._spin_de_recombination = _SigFigSpinBox()
        self._spin_de_recombination.setRange(0.0, 1.0)
        self._spin_de_recombination.setDecimals(3)
        fde.addRow("Recombination (CR):", self._spin_de_recombination)
        self._opt_stack.addWidget(pg_de)        # index 4

        # page 5 – CMA-ES (only present if cma installed)
        if _HAS_CMA:
            pg_cma = QWidget()
            fcma = QFormLayout(pg_cma)
            self._spin_cma_sigma0 = _SigFigSpinBox()
            self._spin_cma_sigma0.setRange(1e-6, 10.0)
            self._spin_cma_sigma0.setDecimals(4)
            self._spin_cma_sigma0.setToolTip(
                "Initial step size in normalised [0,1] parameter space."
            )
            fcma.addRow("Initial σ₀:", self._spin_cma_sigma0)
            self._spin_cma_maxfev = QSpinBox()
            self._spin_cma_maxfev.setRange(10, 100_000)
            self._spin_cma_maxfev.setSingleStep(100)
            fcma.addRow("Max evaluations:", self._spin_cma_maxfev)
            self._opt_stack.addWidget(pg_cma)   # index 4
            _cma_page_idx = 4
        else:
            _cma_page_idx = -1

        # page for Dual Annealing
        pg_da = QWidget()
        fda = QFormLayout(pg_da)
        self._spin_da_maxfev = QSpinBox()
        self._spin_da_maxfev.setRange(10, 100_000)
        self._spin_da_maxfev.setSingleStep(100)
        fda.addRow("Max evaluations:", self._spin_da_maxfev)
        self._spin_da_temp = _SigFigSpinBox()
        self._spin_da_temp.setRange(0.1, 50_000.0)
        self._spin_da_temp.setDecimals(1)
        fda.addRow("Initial temperature:", self._spin_da_temp)
        self._spin_da_restart = _SigFigSpinBox()
        self._spin_da_restart.setRange(1e-10, 1.0)
        self._spin_da_restart.setDecimals(8)
        self._spin_da_restart.setToolTip(
            "Restart temperature ratio (fraction of initial temp)."
        )
        fda.addRow("Restart temp ratio:", self._spin_da_restart)
        _da_page_idx = self._opt_stack.count()
        self._opt_stack.addWidget(pg_da)

        # page for Basin Hopping
        pg_bh = QWidget()
        fbh = QFormLayout(pg_bh)
        self._spin_bh_niter = QSpinBox()
        self._spin_bh_niter.setRange(1, 10_000)
        fbh.addRow("Iterations:", self._spin_bh_niter)
        self._spin_bh_step = _SigFigSpinBox()
        self._spin_bh_step.setRange(1e-6, 10.0)
        self._spin_bh_step.setDecimals(4)
        self._spin_bh_step.setToolTip(
            "Step size for the random displacement in normalised space."
        )
        fbh.addRow("Step size:", self._spin_bh_step)
        _bh_page_idx = self._opt_stack.count()
        self._opt_stack.addWidget(pg_bh)

        # map method name → stack page index
        self._opt_page_map = {
            "Nelder-Mead": 0,
            "Powell": 1,
            "Pattern Search": 2,
            "Newton-CG": 3,
            "Differential Evolution": 4,
            "CMA-ES": _cma_page_idx if _HAS_CMA else 0,
            "Dual Annealing": _da_page_idx,
            "Basin Hopping": _bh_page_idx,
        }

        root.addWidget(self._opt_stack)

        # ── switch stack page when algorithm changes ───────────────────
        def _on_method_changed(text):
            idx = self._opt_page_map.get(text, 0)
            self._opt_stack.setCurrentIndex(idx)
            # workers only useful for DE and Pattern Search
            self._spin_opt_workers.setEnabled(
                text in ("Differential Evolution", "Pattern Search")
            )

        self._cmb_opt_method.currentTextChanged.connect(_on_method_changed)

        # ── buttons ────────────────────────────────────────────────────
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(lambda: (self._opt_dlg_save(), self._save_settings(), dlg.accept()))
        btn_box.rejected.connect(dlg.reject)
        root.addWidget(btn_box)

        self._opt_dlg = dlg

    def _opt_dlg_load(self):
        """Sync GUI widgets from self._opt_config."""
        c = self._opt_config
        idx = self._cmb_opt_method.findText(c.method)
        if idx >= 0:
            self._cmb_opt_method.setCurrentIndex(idx)
        self._spin_opt_workers.setValue(c.n_workers)
        self._spin_opt_maxfev.setValue(c.max_fev)
        self._spin_opt_xtol.setValue(c.x_tol)
        self._spin_opt_ftol.setValue(c.f_tol)
        self._chk_nm_adaptive.setChecked(c.nm_adaptive)
        self._chk_ps_complete.setChecked(c.ps_complete_poll)
        self._spin_ps_mesh_init.setValue(c.ps_initial_mesh)
        self._spin_ps_expand.setValue(c.ps_mesh_expansion)
        self._spin_ps_contract.setValue(c.ps_mesh_contraction)
        self._spin_ps_search.setValue(c.ps_search_pts)
        self._chk_ps_debug.setChecked(c.ps_debug_log)
        self._chk_ncg_debug.setChecked(c.ps_debug_log)
        self._cmb_de_strategy.setCurrentText(c.de_strategy)
        self._spin_de_maxiter.setValue(c.de_maxiter)
        self._spin_de_popsize.setValue(c.de_popsize)
        self._spin_de_mutation.setValue(c.de_mutation)
        self._spin_de_recombination.setValue(c.de_recombination)
        if _HAS_CMA:
            self._spin_cma_sigma0.setValue(c.cma_sigma0)
            self._spin_cma_maxfev.setValue(c.cma_maxfev)
        self._spin_da_maxfev.setValue(c.da_maxfev)
        self._spin_da_temp.setValue(c.da_initial_temp)
        self._spin_da_restart.setValue(c.da_restart_temp)
        self._spin_bh_niter.setValue(c.bh_n_iter)
        self._spin_bh_step.setValue(c.bh_stepsize)
        # trigger page switch
        self._cmb_opt_method.currentTextChanged.emit(c.method)

    def _opt_dlg_save(self):
        """Read GUI widgets back into self._opt_config."""
        c = self._opt_config
        c.method = self._cmb_opt_method.currentText()
        c.n_workers = self._spin_opt_workers.value()
        c.max_fev = self._spin_opt_maxfev.value()
        c.x_tol = self._spin_opt_xtol.value()
        c.f_tol = self._spin_opt_ftol.value()
        c.nm_adaptive = self._chk_nm_adaptive.isChecked()
        c.ps_complete_poll = self._chk_ps_complete.isChecked()
        c.ps_initial_mesh = self._spin_ps_mesh_init.value()
        c.ps_mesh_expansion = self._spin_ps_expand.value()
        c.ps_mesh_contraction = self._spin_ps_contract.value()
        c.ps_search_pts = self._spin_ps_search.value()
        c.ps_debug_log = self._chk_ps_debug.isChecked() or self._chk_ncg_debug.isChecked()
        c.de_strategy = self._cmb_de_strategy.currentText()
        c.de_maxiter = self._spin_de_maxiter.value()
        c.de_popsize = self._spin_de_popsize.value()
        c.de_mutation = self._spin_de_mutation.value()
        c.de_recombination = self._spin_de_recombination.value()
        if _HAS_CMA:
            c.cma_sigma0 = self._spin_cma_sigma0.value()
            c.cma_maxfev = self._spin_cma_maxfev.value()
        c.da_maxfev = self._spin_da_maxfev.value()
        c.da_initial_temp = self._spin_da_temp.value()
        c.da_restart_temp = self._spin_da_restart.value()
        c.bh_n_iter = self._spin_bh_niter.value()
        c.bh_stepsize = self._spin_bh_step.value()

    # =====================================================================
    # settings persistence (JSON)
    # =====================================================================

    def _settings_path(self) -> Path:
        if getattr(sys, "frozen", False):
            # PyInstaller: write next to the exe, not inside the read-only bundle
            return Path(sys.executable).parent / "settings.json"
        return Path(__file__).parent / "settings.json"

    def _collect_parameter_defaults(self) -> dict:
        params = {}
        for name, row in self._param_rows.items():
            factor = self._get_unit_factor(name)
            fw = self._fit_widgets.get(name, {})
            unit_combo = row.get("unit_combo")
            unit_index = unit_combo.currentIndex() if unit_combo is not None else 0
            params[name] = {
                "value_si": float(row["spin"].value()) * factor,
                "unit_index": int(unit_index),
                "fit": bool(fw.get("chk_fit").isChecked()) if fw.get("chk_fit") else True,
                "lb_si": float(fw["spin_lb"].value()) * factor if fw.get("spin_lb") else None,
                "ub_si": float(fw["spin_ub"].value()) * factor if fw.get("spin_ub") else None,
                "scale": fw["cmb_scale"].currentText() if fw.get("cmb_scale") else "lin",
            }
        return params

    def _apply_parameter_defaults(self, params: dict):
        for name, saved in params.items():
            row = self._param_rows.get(name)
            if not row:
                continue

            unit_combo = row.get("unit_combo")
            unit_options = row.get("unit_options") or []
            unit_index = int(saved.get("unit_index", 0))
            if unit_combo is not None and 0 <= unit_index < len(unit_options):
                unit_combo.blockSignals(True)
                unit_combo.setCurrentIndex(unit_index)
                unit_combo.blockSignals(False)
                row["unit_index"] = unit_index

            factor = self._get_unit_factor(name)
            if factor == 0:
                factor = 1.0

            if "value_si" in saved:
                row["spin"].setValue(float(saved["value_si"]) / factor)

            fw = self._fit_widgets.get(name)
            if fw:
                if "fit" in saved:
                    fw["chk_fit"].setChecked(bool(saved["fit"]))
                if "lb_si" in saved and saved["lb_si"] is not None:
                    fw["spin_lb"].setValue(float(saved["lb_si"]) / factor)
                if "ub_si" in saved and saved["ub_si"] is not None:
                    fw["spin_ub"].setValue(float(saved["ub_si"]) / factor)
                if "scale" in saved:
                    idx = fw["cmb_scale"].findText(str(saved["scale"]))
                    if idx >= 0:
                        fw["cmb_scale"].setCurrentIndex(idx)

    @staticmethod
    def _normalise_ui_defaults(ui: dict) -> dict:
        if not isinstance(ui, dict):
            return {}

        active_model = ui.get("active_model", ui.get("model"))
        models = ui.get("models", {})
        if not isinstance(models, dict):
            models = {}
        else:
            models = dict(models)

        # Backwards compatibility with the first single-model settings schema.
        if "parameters" in ui and ui.get("model") in load_available_models():
            model_key = ui["model"]
            old_model = dict(models.get(model_key, {}))
            old_model["parameters"] = ui.get("parameters", {})
            models[model_key] = old_model

        out = {
            "active_model": active_model,
            "Req_um": ui.get("Req_um"),
            "tspan_us": ui.get("tspan_us"),
            "fit_window_auto": ui.get("fit_window_auto"),
            "fit_window_cycles": ui.get("fit_window_cycles"),
            "models": models,
        }
        return out

    @classmethod
    def _normalise_import_wizard_keywords(cls, data: dict) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        source = data if isinstance(data, dict) else {}
        for field, defaults in cls.DEFAULT_IMPORT_WIZARD_KEYWORDS.items():
            values = source.get(field, defaults)
            if not isinstance(values, (list, tuple)):
                values = defaults
            cleaned: list[str] = []
            for value in values:
                text = str(value).strip()
                if text and text not in cleaned:
                    cleaned.append(text)
            out[field] = cleaned or list(defaults)
        return out

    @classmethod
    def _normalise_import_wizard_units(cls, data: dict) -> dict[str, str]:
        out = dict(cls.DEFAULT_IMPORT_WIZARD_UNITS)
        source = data if isinstance(data, dict) else {}
        valid = {
            "t_exp": {"s", "us"},
            "R_exp": {"m", "um", "pixel"},
            "t_sim": {"s", "us"},
            "R_sim": {"m", "um"},
            "Req": {"m", "um"},
            "Rmax": {"m", "um"},
        }
        for name, allowed in valid.items():
            unit = str(source.get(name, out.get(name, "")))
            if unit in allowed:
                out[name] = unit
        return out

    def _parameter_defaults_for_model(self, model_key: str) -> dict:
        ui = self._normalise_ui_defaults(self._saved_ui_defaults)
        model_data = ui.get("models", {}).get(model_key, {})
        params = model_data.get("parameters", {})
        return params if isinstance(params, dict) else {}

    def _remember_current_model_parameters(self) -> tuple[str | None, dict]:
        if self._current_model is None or not self._param_rows:
            return None, {}
        model_key = self._current_model.id
        snapshot = self._collect_parameter_defaults()
        self._model_param_memory[model_key] = snapshot
        return model_key, snapshot

    def _parameter_state_for_model_switch(
        self,
        target_model: ConstitutiveModel,
        inherited_params: dict,
        source_model_key: str | None,
    ) -> dict:
        target_names = {p.name for p in target_model.parameters}
        state: dict = {}

        saved = self._parameter_defaults_for_model(target_model.id)
        if isinstance(saved, dict):
            state.update(saved)

        remembered = self._model_param_memory.get(target_model.id)
        if isinstance(remembered, dict):
            state.update(remembered)

        for name, value in inherited_params.items():
            if name in target_names:
                state[name] = value

        alias_map = self.PARAM_INHERIT_ALIASES.get(target_model.id, {})
        if source_model_key is not None and alias_map:
            for target_name, source_name in alias_map.items():
                if target_name in target_names and source_name in inherited_params:
                    state[target_name] = inherited_params[source_name]

        return state

    def _save_settings(self, show_status: bool = False, include_ui: bool = False):
        c = self._opt_config
        try:
            existing = json.loads(self._settings_path().read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        existing_ui = self._normalise_ui_defaults(existing.get("ui", {})) if isinstance(existing, dict) else {}

        data = {
            "physics": {
                "bubble_model": self._bubble_model,
                "P_inf":   float(self.spin_P_inf.value()),
                "rho":     float(self.spin_rho.value()),
                "c_long":  float(self.spin_c_long.value()),
                "gamma":   float(self.spin_gamma.value()),
                "NT":      int(self.spin_NT.value()),
                "solver":  self._cmb_solver.currentText(),
                "rtol":    self.le_rtol.text(),
                "atol":    self.le_atol.text(),
            },
            "optimizer": {
                "method":             c.method,
                "n_workers":          c.n_workers,
                "max_fev":            c.max_fev,
                "x_tol":              c.x_tol,
                "f_tol":              c.f_tol,
                "nm_adaptive":        c.nm_adaptive,
                "de_strategy":        c.de_strategy,
                "de_maxiter":         c.de_maxiter,
                "de_popsize":         c.de_popsize,
                "de_mutation":        c.de_mutation,
                "de_recombination":   c.de_recombination,
                "cma_sigma0":         c.cma_sigma0,
                "cma_maxfev":         c.cma_maxfev,
                "da_maxfev":          c.da_maxfev,
                "da_initial_temp":    c.da_initial_temp,
                "da_restart_temp":    c.da_restart_temp,
                "bh_n_iter":          c.bh_n_iter,
                "bh_stepsize":        c.bh_stepsize,
                "ps_complete_poll":   c.ps_complete_poll,
                "ps_mesh_contraction": c.ps_mesh_contraction,
                "ps_mesh_expansion":  c.ps_mesh_expansion,
                "ps_initial_mesh":    c.ps_initial_mesh,
                "ps_search_pts":      c.ps_search_pts,
                "ps_debug_log":       c.ps_debug_log,
            },
            "job_list": {
                "success_threshold_enabled": self._job_success_threshold_enabled,
                "success_lsqerr": self._job_success_lsqerr,
                "output_dir": self._job_output_dir,
                "ask_output_dir": self._job_ask_output_dir,
                "chain_best_fit_initial": self._job_chain_best_fit_initial,
                "parallel_workers": self._job_parallel_workers,
                "import_name_cleanup_enabled": self._job_import_name_cleanup_enabled,
                "import_name_cleanup_regex": self._job_import_name_cleanup_regex,
            },
            "import_wizard": {
                "keywords": self._normalise_import_wizard_keywords(self._import_wizard_keywords),
                "units": self._normalise_import_wizard_units(self._import_wizard_units),
                "um_per_pixel": float(self._import_wizard_um_per_pixel),
                "fps": float(self._import_wizard_fps),
                "remove_negative_R": bool(self._import_wizard_remove_below),
                "remove_isolated_spikes": bool(self._import_wizard_remove_spikes),
                "spike_threshold": float(self._import_wizard_spike_threshold),
            },
        }
        if include_ui:
            current_model = self._get_active_model_key()
            models = dict(existing_ui.get("models", {}))
            model_data = dict(models.get(current_model, {}))
            model_data["parameters"] = self._collect_parameter_defaults()
            models[current_model] = model_data
            data["ui"] = {
                "active_model": current_model,
                "Req_um": float(self.spin_Req_um.value()),
                "tspan_us": float(self.spin_tspan_us.value()),
                "fit_window_auto": bool(self.chk_fit_window_cycles.isChecked()),
                "fit_window_cycles": int(self.spin_fit_window_cycles.value()),
                "models": models,
            }
            self._saved_ui_defaults = data["ui"]
        elif existing_ui:
            data["ui"] = existing_ui

        try:
            self._settings_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
            if show_status:
                self.statusBar().showMessage(f"Default settings saved to {self._settings_path()}")
        except Exception:
            if show_status:
                QMessageBox.warning(self, "Save failed", "Could not save default settings.")
            pass

    def _load_settings(self):
        try:
            text = self._settings_path().read_text(encoding="utf-8")
            data = json.loads(text)
        except Exception:
            return

        ui = self._normalise_ui_defaults(data.get("ui", {}))
        self._saved_ui_defaults = ui
        model_key = ui.get("active_model")
        if model_key in self._available_models:
            idx = self._cmb_model.findText(model_key)
            if idx >= 0:
                self._cmb_model.setCurrentIndex(idx)

        phys = data.get("physics", {})
        if "bubble_model" in phys:
            bm = phys["bubble_model"]
            self._bubble_model = bm
            self._act_rp.setChecked(bm == "Rayleigh-Plesset")
            self._act_km.setChecked(bm != "Rayleigh-Plesset")
        if "P_inf"  in phys: self.spin_P_inf.setValue(float(phys["P_inf"]))
        if "rho"    in phys: self.spin_rho.setValue(float(phys["rho"]))
        if "c_long" in phys: self.spin_c_long.setValue(float(phys["c_long"]))
        if "gamma"  in phys: self.spin_gamma.setValue(float(phys["gamma"]))
        if "NT"     in phys: self.spin_NT.setValue(int(phys["NT"]))
        if "solver" in phys:
            idx = self._cmb_solver.findText(phys["solver"])
            if idx >= 0:
                self._cmb_solver.setCurrentIndex(idx)
        if "rtol" in phys: self.le_rtol.setText(phys["rtol"])
        if "atol" in phys: self.le_atol.setText(phys["atol"])

        opt = data.get("optimizer", {})
        c = self._opt_config
        if "method"             in opt: c.method             = opt["method"]
        if "n_workers"          in opt: c.n_workers          = int(opt["n_workers"])
        if "max_fev"            in opt: c.max_fev            = int(opt["max_fev"])
        if "x_tol"              in opt: c.x_tol              = float(opt["x_tol"])
        if "f_tol"              in opt: c.f_tol              = float(opt["f_tol"])
        if "nm_adaptive"        in opt: c.nm_adaptive        = bool(opt["nm_adaptive"])
        if "de_strategy"        in opt: c.de_strategy        = opt["de_strategy"]
        if "de_maxiter"         in opt: c.de_maxiter         = int(opt["de_maxiter"])
        if "de_popsize"         in opt: c.de_popsize         = int(opt["de_popsize"])
        if "de_mutation"        in opt: c.de_mutation        = float(opt["de_mutation"])
        if "de_recombination"   in opt: c.de_recombination   = float(opt["de_recombination"])
        if "cma_sigma0"         in opt: c.cma_sigma0         = float(opt["cma_sigma0"])
        if "cma_maxfev"         in opt: c.cma_maxfev         = int(opt["cma_maxfev"])
        if "da_maxfev"          in opt: c.da_maxfev          = int(opt["da_maxfev"])
        if "da_initial_temp"    in opt: c.da_initial_temp    = float(opt["da_initial_temp"])
        if "da_restart_temp"    in opt: c.da_restart_temp    = float(opt["da_restart_temp"])
        if "bh_n_iter"          in opt: c.bh_n_iter          = int(opt["bh_n_iter"])
        if "bh_stepsize"        in opt: c.bh_stepsize        = float(opt["bh_stepsize"])
        if "ps_complete_poll"   in opt: c.ps_complete_poll   = bool(opt["ps_complete_poll"])
        if "ps_mesh_contraction" in opt: c.ps_mesh_contraction = float(opt["ps_mesh_contraction"])
        if "ps_mesh_expansion"  in opt: c.ps_mesh_expansion  = float(opt["ps_mesh_expansion"])
        if "ps_initial_mesh"    in opt: c.ps_initial_mesh    = float(opt["ps_initial_mesh"])
        if "ps_search_pts"      in opt: c.ps_search_pts      = int(opt["ps_search_pts"])
        if "ps_debug_log"       in opt: c.ps_debug_log       = bool(opt["ps_debug_log"])

        job_list = data.get("job_list", {})
        if "success_threshold_enabled" in job_list:
            self._job_success_threshold_enabled = bool(job_list["success_threshold_enabled"])
        if "success_lsqerr" in job_list:
            self._job_success_lsqerr = float(job_list["success_lsqerr"])
        if "output_dir" in job_list:
            self._job_output_dir = str(job_list["output_dir"])
        if "ask_output_dir" in job_list:
            self._job_ask_output_dir = bool(job_list["ask_output_dir"])
        if "chain_best_fit_initial" in job_list:
            self._job_chain_best_fit_initial = bool(job_list["chain_best_fit_initial"])
        if "parallel_workers" in job_list:
            self._job_parallel_workers = max(1, int(job_list["parallel_workers"]))
        if "import_name_cleanup_enabled" in job_list:
            self._job_import_name_cleanup_enabled = bool(job_list["import_name_cleanup_enabled"])
        if "import_name_cleanup_regex" in job_list:
            self._job_import_name_cleanup_regex = str(job_list["import_name_cleanup_regex"])

        import_wizard = data.get("import_wizard", {})
        if isinstance(import_wizard, dict):
            self._import_wizard_keywords = self._normalise_import_wizard_keywords(
                import_wizard.get("keywords", {})
            )
            self._import_wizard_units = self._normalise_import_wizard_units(
                import_wizard.get("units", {})
            )
            if "um_per_pixel" in import_wizard:
                self._import_wizard_um_per_pixel = float(import_wizard["um_per_pixel"])
            if "fps" in import_wizard:
                self._import_wizard_fps = float(import_wizard["fps"])
            if "remove_negative_R" in import_wizard:
                self._import_wizard_remove_below = bool(import_wizard["remove_negative_R"])
            elif "remove_below_threshold" in import_wizard:
                self._import_wizard_remove_below = bool(import_wizard["remove_below_threshold"])
            if "remove_isolated_spikes" in import_wizard:
                self._import_wizard_remove_spikes = bool(import_wizard["remove_isolated_spikes"])
            if "spike_threshold" in import_wizard:
                self._import_wizard_spike_threshold = float(import_wizard["spike_threshold"])

        if ui.get("Req_um") is not None:
            self.spin_Req_um.setValue(float(ui["Req_um"]))
        if ui.get("tspan_us") is not None:
            self.spin_tspan_us.setValue(float(ui["tspan_us"]))
        if ui.get("fit_window_cycles") is not None:
            self.spin_fit_window_cycles.setValue(int(ui["fit_window_cycles"]))
        if ui.get("fit_window_auto") is not None:
            self.chk_fit_window_cycles.setChecked(bool(ui["fit_window_auto"]))
        self._apply_parameter_defaults(self._parameter_defaults_for_model(self._get_active_model_key()))
        if self.chk_fit_window_cycles.isChecked():
            self._apply_fit_window_cycles()

    def on_save_defaults(self):
        self._save_settings(show_status=True, include_ui=True)

    # =====================================================================
    # dynamic parameter panel
    # =====================================================================

    def _on_model_changed(self, model_key: str):
        loader = self._available_models.get(model_key)
        if not loader:
            return
        old_model_key, inherited_params = self._remember_current_model_parameters()
        model = loader()
        self._current_model = model
        self._model_constants = model.constants
        self._param_box.setTitle(f"Parameters ({model.display_name})")
        self._rebuild_param_panel(model)
        if old_model_key is None:
            params_to_apply = self._parameter_defaults_for_model(model_key)
        else:
            params_to_apply = self._parameter_state_for_model_switch(
                model, inherited_params, old_model_key
            )
        self._apply_parameter_defaults(params_to_apply)
        self._set_mode(self.state.mode)  # refresh fit-control visibility
        # GMOD models require much tighter ODE tolerances than NHKV.
        if model_key in ("GMOD1", "GMOD2"):
            self.le_rtol.setText("1e-9")
            self.le_atol.setText("1e-9")
        else:
            self.le_rtol.setText("1e-8")
            self.le_atol.setText("1e-7")
        if self.chk_fit_window_cycles.isChecked():
            self._apply_fit_window_cycles()

    def _rebuild_param_panel(self, model: ConstitutiveModel):
        # tear down existing rows
        self._param_rows.clear()
        self._fit_widgets.clear()
        old_content = self._param_scroll.takeWidget()
        if old_content is not None:
            old_content.deleteLater()

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(2, 2, 2, 2)

        for i, p in enumerate(model.parameters):
            self._add_param_row(p, lay)
            if i < len(model.parameters) - 1:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setFrameShadow(QFrame.Shadow.Sunken)
                lay.addWidget(sep)

        lay.addStretch(1)
        self._param_scroll.setWidget(content)

    def _add_param_row(self, p: ConstitutiveParameter, parent_layout: QVBoxLayout):
        # Column widths — kept identical between both rows for alignment:
        _W_LABEL = 46   # name label  ↔  Fit checkbox
        _W_SPIN  = 82   # value spin  ↔  lb spin
        _W_UNIT  = 68   # unit widget (shared column)
        _W_SCALE = 60   # scale combo
        _W_UB    = 16   # "ub" mini-label
        _W_UB_SPIN = 68  # ub spinbox

        # --- value row: [name | value_spin | unit | scale] ---
        row = QHBoxLayout()
        row.setSpacing(4)
        row.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(p.label)
        lbl.setFixedWidth(_W_LABEL)
        row.addWidget(lbl)

        spin = _SigFigSpinBox()
        spin.setRange(-1e15, 1e15)
        spin.setDecimals(10)
        spin.setValue(p.default)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setFixedWidth(_W_SPIN)
        row.addWidget(spin)

        unit_combo: QComboBox | None = None
        unit_options = list(p.units) if p.units else []
        if len(unit_options) > 1:
            unit_combo = _NoWheelComboBox()
            for u in unit_options:
                unit_combo.addItem(u.label)
            unit_combo.setFixedWidth(_W_UNIT)
            row.addWidget(unit_combo)
            unit_combo.currentIndexChanged.connect(
                lambda idx, name=p.name: self._on_unit_changed(name, idx)
            )
        elif unit_options:
            lbl_unit = QLabel(unit_options[0].label)
            lbl_unit.setFixedWidth(_W_UNIT)
            row.addWidget(lbl_unit)

        cmb_scale = _NoWheelComboBox()
        cmb_scale.addItems(["lin", "log"])
        if p.scale == "log":
            cmb_scale.setCurrentIndex(1)
        cmb_scale.setFixedWidth(_W_SCALE)
        row.addWidget(cmb_scale)

        row.addStretch(1)
        parent_layout.addLayout(row)

        # --- fit controls row: [Fit | lb_spin | "ub" | ub_spin] ---
        # Fit checkbox has the same fixed width as the name label above, so
        # lb_spin starts at the exact same x-offset as value_spin. No "lb"
        # label is needed — position makes it unambiguous.
        fw = QWidget()
        flay = QHBoxLayout(fw)
        flay.setContentsMargins(0, 0, 0, 0)
        flay.setSpacing(4)

        chk = QCheckBox("Fit")
        chk.setChecked(p.fit_default)
        chk.setFixedWidth(_W_LABEL)          # ← matches name label width
        flay.addWidget(chk)

        spin_lb = _SigFigSpinBox()
        spin_lb.setRange(-1e15, 1e15)
        spin_lb.setDecimals(10)
        spin_lb.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin_lb.setValue(p.lb)
        spin_lb.setFixedWidth(_W_SPIN)       # ← matches value spin width
        spin_lb.setToolTip("Lower bound (lb)")
        flay.addWidget(spin_lb)

        lbl_ub = QLabel("ub")
        lbl_ub.setFixedWidth(_W_UB)
        lbl_ub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        flay.addWidget(lbl_ub)

        spin_ub = _SigFigSpinBox()
        spin_ub.setRange(-1e15, 1e15)
        spin_ub.setDecimals(10)
        spin_ub.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin_ub.setValue(p.ub)
        spin_ub.setFixedWidth(_W_UB_SPIN)
        spin_ub.setToolTip("Upper bound (ub)")
        flay.addWidget(spin_ub)

        flay.addStretch(1)
        parent_layout.addWidget(fw)

        # store references
        entry = {
            "spin": spin,
            "unit_combo": unit_combo,
            "unit_options": unit_options,
            "unit_index": 0,
            "meta": p,
        }
        self._param_rows[p.name] = entry
        self._fit_widgets[p.name] = {
            "chk_fit": chk,
            "spin_lb": spin_lb,
            "spin_ub": spin_ub,
            "cmb_scale": cmb_scale,
            "row_widget": fw,
        }

    # =====================================================================
    # mode switching
    # =====================================================================

    def _on_primary_action(self):
        if self.state.mode == "fitting":
            self.on_fit()
        else:
            self.on_simulate()

    def _set_mode(self, mode: str):
        if self._fit_worker is not None and self._fit_worker.isRunning():
            if not self._fit_worker._stop_requested:
                QMessageBox.warning(
                    self, "Fitting in progress",
                    "Cannot switch mode while fitting is running. Stop the fit first.",
                )
                self._act_sim.setChecked(self.state.mode == "simulation")
                self._act_fit_mode.setChecked(self.state.mode == "fitting")
                self._act_jobs_mode.setChecked(self.state.mode == "jobs")
                return

        previous_mode = self.state.mode
        self.state.mode = mode
        is_fitting = mode == "fitting"
        is_jobs = mode == "jobs"
        if previous_mode == "jobs" and not is_jobs and not self._loading_job_to_editor:
            self._clear_job_preview_state()

        for fw in self._fit_widgets.values():
            fw["row_widget"].setEnabled(is_fitting)
            fw["cmb_scale"].setEnabled(is_fitting)

        if is_fitting and self.spin_fit_window_cycles.value() < 1:
            self.spin_fit_window_cycles.blockSignals(True)
            self.spin_fit_window_cycles.setValue(1)
            self.spin_fit_window_cycles.blockSignals(False)

        self._fit_window_widget.setVisible(not is_jobs)
        if hasattr(self, "_left_stack"):
            self._left_stack.setCurrentIndex(1 if is_jobs else 0)
        self.btn_primary_action.setText("Fit" if is_fitting else "Simulate")
        self.btn_primary_action.setEnabled(not is_jobs)
        self.btn_add_job.setEnabled(is_fitting or mode == "simulation")
        if is_fitting:
            self.btn_add_job.setText("Update job" if self._editing_job_index is not None else "Add to job list")
            self.btn_add_job.setToolTip("Update the loaded job." if self._editing_job_index is not None else "Add the current fitting setup as a queued job.")
        elif mode == "simulation":
            self.btn_add_job.setText("Add to job list")
            self.btn_add_job.setToolTip("Add the current simulation setup as a queued job.")
        else:
            self.btn_add_job.setText("Add to job list")
            self.btn_add_job.setToolTip("Switch to Simulation or Fitting mode to add a job.")
        self._act_load_exp.setEnabled(not is_jobs)
        self._act_load_params.setEnabled(not is_jobs)
        self._act_save_params.setEnabled(not is_jobs)
        self._act_export_result.setEnabled(not is_jobs)
        self._act_batch_add_jobs.setEnabled(is_fitting or mode == "simulation")
        self._act_batch_create_sim_jobs.setEnabled(mode == "simulation")

        self._act_sim.setChecked(mode == "simulation")
        self._act_fit_mode.setChecked(is_fitting)
        self._act_jobs_mode.setChecked(is_jobs)
        self.canvas.set_drag_callback(None if is_jobs else self._on_fit_window_dragged)
        if is_jobs and self._jobs and self._selected_job_index() is None:
            self.tbl_jobs.selectRow(0)

        self._redraw_all()

    # =====================================================================
    # helpers
    # =====================================================================

    @staticmethod
    def _sec_to_hms(x: float | None) -> str:
        if x is None or not np.isfinite(x) or x < 0:
            return "--:--:--"
        x = int(round(x))
        hh = x // 3600
        mm = (x % 3600) // 60
        ss = x % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    def _update_view_buttons(self):
        is_dim = self.state.view_mode == "dimensional"
        self.btn_dim.setEnabled(not is_dim)
        self.btn_norm.setEnabled(is_dim)

    def _update_window_title(self):
        exp_path = self.state.exp_path
        if exp_path:
            name = Path(exp_path).name
        else:
            name = ""
        self.setWindowTitle(f"{self.APP_TITLE} — {name}" if name else self.APP_TITLE)

    def _clear_job_preview_state(self):
        self._job_preview_fit_t = None
        self._job_preview_fit_R = None
        self._job_preview_fit_meta = None
        self._job_preview_window = None
        self.state.best_fit_t = None
        self.state.best_fit_R = None
        self.state.best_fit_meta = None

    def _get_unit_factor(self, param_name: str) -> float:
        row = self._param_rows.get(param_name)
        if not row or not row["unit_options"]:
            return 1.0
        combo = row["unit_combo"]
        idx = combo.currentIndex() if combo is not None else 0
        if 0 <= idx < len(row["unit_options"]):
            return row["unit_options"][idx].factor
        return 1.0

    def _get_param_si(self) -> dict[str, float]:
        result = {}
        for name, row in self._param_rows.items():
            result[name] = float(row["spin"].value()) * self._get_unit_factor(name)
        return result

    def _collect_fit_setup(self) -> tuple[dict[str, bool], dict[str, str], dict[str, tuple[float, float]]]:
        fit_flags: dict[str, bool] = {}
        scales: dict[str, str] = {}
        bounds_si: dict[str, tuple[float, float]] = {}
        for name in self._param_rows:
            fw = self._fit_widgets.get(name)
            if not fw:
                continue
            fit_flags[name] = bool(fw["chk_fit"].isChecked())
            scales[name] = fw["cmb_scale"].currentText()
            factor = self._get_unit_factor(name)
            lb = float(fw["spin_lb"].value()) * factor
            ub = float(fw["spin_ub"].value()) * factor
            if lb > ub:
                lb, ub = ub, lb
            bounds_si[name] = (lb, ub)
        return fit_flags, scales, bounds_si

    def _get_active_model_key(self) -> str:
        return self._cmb_model.currentText()

    # --- build model-specific inputs ---

    def _get_model_definition(self, model_key: str | None = None) -> ConstitutiveModel | None:
        key = model_key or self._get_active_model_key()
        if self._current_model is not None and self._current_model.id == key:
            return self._current_model
        loader = self._available_models.get(key)
        if loader is None:
            return None
        try:
            return loader()
        except Exception:
            return None

    def _plugin_entrypoint_for_model(self, model_key: str | None = None) -> str:
        model = self._get_model_definition(model_key)
        return str(model.solver_entrypoint or "") if model is not None else ""

    def _get_solver_settings(self) -> dict:
        try:
            rtol = float(self.le_rtol.text())
        except ValueError:
            rtol = 1e-8
        try:
            atol = float(self.le_atol.text())
        except ValueError:
            atol = 1e-7
        return dict(
            solver_method=self._cmb_solver.currentText(),
            rel_tol=rtol,
            abs_tol=atol,
        )

    def _plugin_context(
        self,
        *,
        model_key: str,
        Req: float,
        NT: int,
        P_inf: float,
        rho: float,
        const: dict,
        solver: dict,
        Rmax_exp: float = 0.0,
    ) -> dict:
        return {
            "model_key": model_key,
            "Req": float(Req),
            "NT": int(NT),
            "P_inf": float(P_inf),
            "rho": float(rho),
            "gamma": float(self.spin_gamma.value()),
            "c_long": float(self.spin_c_long.value()),
            "bubble_model": self._bubble_model,
            "constants": dict(const),
            "solver": dict(solver),
            "Rmax_exp": float(Rmax_exp),
        }

    def _build_sim_inputs(self, params: dict[str, float]):
        """Build solver inputs from GUI params + constants for the current model."""
        key = self._get_active_model_key()
        const = dict(self._model_constants)
        const["c_long"] = float(self.spin_c_long.value())
        const["gamma"] = float(self.spin_gamma.value())
        Req = float(self.spin_Req_um.value()) * 1e-6
        tspan = float(self.spin_tspan_us.value()) * 1e-6
        NT = int(self.spin_NT.value())
        solver = self._get_solver_settings()
        P_inf = float(self.spin_P_inf.value())
        rho = float(self.spin_rho.value())

        bm = self._bubble_model
        plugin_entrypoint = self._plugin_entrypoint_for_model(key)
        if plugin_entrypoint:
            Rmax_exp = (
                find_rmax_value(self.state.exp_t, self.state.exp_R)
                if self.state.exp_t is not None and self.state.exp_R is not None
                else 0.0
            )
            return {
                "solver_entrypoint": plugin_entrypoint,
                "params_si": dict(params),
                "tspan": tspan,
                "context": self._plugin_context(
                    model_key=key, Req=Req, NT=NT, P_inf=P_inf, rho=rho,
                    const=const, solver=solver, Rmax_exp=Rmax_exp,
                ),
            }
        if key == "NHKV":
            const_kw = {k: v for k, v in const.items()
                        if k in NhkvInputs.__dataclass_fields__}
            return NhkvInputs(
                U0=params["U0"], G=params["G"], mu=params["mu"],
                Req=Req, tspan=tspan, NT=NT,
                P_inf=P_inf, rho=rho, bubble_model=bm, **solver, **const_kw,
            )
        elif key == "NHKV (Rmax)":
            Rmax_exp = (
                find_rmax_value(self.state.exp_t, self.state.exp_R)
                if self.state.exp_t is not None and self.state.exp_R is not None
                else Req
            )
            const_kw = {k: v for k, v in const.items()
                        if k in NhkvRmaxInputs.__dataclass_fields__}
            return NhkvRmaxInputs(
                G=params["G"], mu=params["mu"],
                Req=Req, Rmax_exp=Rmax_exp, tspan=tspan, NT=NT,
                P_inf=P_inf, rho=rho, bubble_model=bm, **solver, **const_kw,
            )
        elif key == "GMOD1":
            const_kw = {k: v for k, v in const.items()
                        if k in GMOD1Inputs.__dataclass_fields__}
            return GMOD1Inputs(
                U0=params.get("U0", 100.0),
                GA=params.get("GA", 8e6),
                alpha=params.get("alpha", 1.0),
                GB=params.get("GB", 1e4),
                beta=params.get("beta", 1.0),
                mu=params.get("mu", 0.226),
                lambda_Y=params.get("lambda_Y", 1.5),
                Req=Req, tspan=tspan, NT=NT,
                P_inf=P_inf, rho=rho, bubble_model=bm, **solver, **const_kw,
            )
        else:  # GMOD2
            const_kw = {k: v for k, v in const.items()
                        if k in GMODInputs.__dataclass_fields__}
            return GMODInputs(
                U0=params.get("U0", 100.0),
                GA1=params.get("GA1", 8e6),
                GA2=params.get("GA2", 1e-10),
                alpha1=params.get("alpha1", 1.0),
                alpha2=params.get("alpha2", 1.0),
                GB1=params.get("GB1", 1e4),
                GB2=params.get("GB2", 1e-10),
                beta1=params.get("beta1", 1.0),
                beta2=params.get("beta2", 1.0),
                mu=params.get("mu", 0.226),
                lambda_Y=params.get("lambda_Y", 1.5),
                Req=Req, tspan=tspan, NT=NT,
                P_inf=P_inf, rho=rho, bubble_model=bm, **solver, **const_kw,
            )

    def _get_simulate_fn(self):
        key = self._get_active_model_key()
        if self._plugin_entrypoint_for_model(key):
            return _run_plugin_simulation
        if key == "NHKV":
            return simulate_nhkv_lic
        if key == "NHKV (Rmax)":
            return simulate_nhkv_rmax_lic
        if key == "GMOD1":
            return simulate_gmod1_lic
        return simulate_gmod_lic  # GMOD2

    def _make_sim_for_fit(self, params_si: dict, tspan: float):
        """Called by the fitting engine to run one simulation."""
        key = self._get_active_model_key()
        const = dict(self._model_constants)
        const["c_long"] = float(self.spin_c_long.value())
        const["gamma"] = float(self.spin_gamma.value())
        Req = float(self.spin_Req_um.value()) * 1e-6
        NT = int(self.spin_NT.value())
        solver = self._get_solver_settings()
        P_inf = float(self.spin_P_inf.value())
        rho = float(self.spin_rho.value())

        bm = self._bubble_model
        plugin_entrypoint = self._plugin_entrypoint_for_model(key)
        if plugin_entrypoint:
            Rmax_exp = (
                find_rmax_value(self.state.exp_t, self.state.exp_R)
                if self.state.exp_t is not None and self.state.exp_R is not None
                else 0.0
            )
            ctx = self._plugin_context(
                model_key=key, Req=Req, NT=NT, P_inf=P_inf, rho=rho,
                const=const, solver=solver, Rmax_exp=Rmax_exp,
            )
            return _plugin_sim_call(_PluginSimSpec(plugin_entrypoint, ctx), params_si, tspan)
        if key == "NHKV":
            const_kw = {k: v for k, v in const.items()
                        if k in NhkvInputs.__dataclass_fields__}
            inp = NhkvInputs(
                U0=params_si["U0"], G=params_si["G"], mu=params_si["mu"],
                Req=Req, tspan=tspan, NT=NT,
                P_inf=P_inf, rho=rho, bubble_model=bm, **solver, **const_kw,
            )
            return simulate_nhkv_lic(inp)
        elif key == "NHKV (Rmax)":
            Rmax_exp = (
                find_rmax_value(self.state.exp_t, self.state.exp_R)
                if self.state.exp_t is not None and self.state.exp_R is not None
                else Req
            )
            const_kw = {k: v for k, v in const.items()
                        if k in NhkvRmaxInputs.__dataclass_fields__}
            inp = NhkvRmaxInputs(
                G=params_si["G"], mu=params_si["mu"],
                Req=Req, Rmax_exp=Rmax_exp, tspan=tspan, NT=NT,
                P_inf=P_inf, rho=rho, bubble_model=bm, **solver, **const_kw,
            )
            return simulate_nhkv_rmax_lic(inp)
        elif key == "GMOD1":
            const_kw = {k: v for k, v in const.items()
                        if k in GMOD1Inputs.__dataclass_fields__}
            inp = GMOD1Inputs(
                U0=params_si.get("U0", 100.0),
                GA=params_si.get("GA", 8e6),
                alpha=params_si.get("alpha", 1.0),
                GB=params_si.get("GB", 1e4),
                beta=params_si.get("beta", 1.0),
                mu=params_si.get("mu", 0.226),
                lambda_Y=params_si.get("lambda_Y", 1.5),
                Req=Req, tspan=tspan, NT=NT,
                P_inf=P_inf, rho=rho, bubble_model=bm, **solver, **const_kw,
            )
            return simulate_gmod1_lic(inp)
        else:  # GMOD2
            const_kw = {k: v for k, v in const.items()
                        if k in GMODInputs.__dataclass_fields__}
            inp = GMODInputs(
                U0=params_si.get("U0", 100.0),
                GA1=params_si.get("GA1", 8e6),
                GA2=params_si.get("GA2", 1e-10),
                alpha1=params_si.get("alpha1", 1.0),
                alpha2=params_si.get("alpha2", 1.0),
                GB1=params_si.get("GB1", 1e4),
                GB2=params_si.get("GB2", 1e-10),
                beta1=params_si.get("beta1", 1.0),
                beta2=params_si.get("beta2", 1.0),
                mu=params_si.get("mu", 0.226),
                lambda_Y=params_si.get("lambda_Y", 1.5),
                Req=Req, tspan=tspan, NT=NT,
                P_inf=P_inf, rho=rho, bubble_model=bm, **solver, **const_kw,
            )
            return simulate_gmod_lic(inp)

    # --- time conversions ---

    def _time_s_to_view(self, t_s: float) -> float:
        if self.state.view_mode == "dimensional":
            return t_s * 1e6
        P_inf = float(self.spin_P_inf.value())
        rho = float(self.spin_rho.value())
        R_eq = float(self.spin_Req_um.value()) * 1e-6
        Uc = np.sqrt(P_inf / rho) if rho > 0 else 1.0
        tc = R_eq / Uc if Uc > 0 else 1.0
        t_rmax = 0.0
        if self.state.exp_t is not None and self.state.exp_R is not None:
            t_rmax = float(self.state.exp_t[int(np.argmax(self.state.exp_R))])
        return (t_s - t_rmax) / tc

    def _time_view_to_s(self, t_view: float) -> float:
        if self.state.view_mode == "dimensional":
            return t_view * 1e-6
        P_inf = float(self.spin_P_inf.value())
        rho = float(self.spin_rho.value())
        R_eq = float(self.spin_Req_um.value()) * 1e-6
        Uc = np.sqrt(P_inf / rho) if rho > 0 else 1.0
        tc = R_eq / Uc if Uc > 0 else 1.0
        t_rmax = 0.0
        if self.state.exp_t is not None and self.state.exp_R is not None:
            t_rmax = float(self.state.exp_t[int(np.argmax(self.state.exp_R))])
        return t_view * tc + t_rmax

    # =====================================================================
    # unit-change handler (generic)
    # =====================================================================

    def _on_unit_changed(self, param_name: str, new_index: int):
        row = self._param_rows.get(param_name)
        if not row or not row["unit_options"]:
            return
        old_index = row.get("unit_index", 0)
        if new_index < 0 or new_index >= len(row["unit_options"]):
            return
        old_factor = row["unit_options"][old_index].factor
        new_factor = row["unit_options"][new_index].factor

        val_si = float(row["spin"].value()) * old_factor
        row["spin"].blockSignals(True)
        row["spin"].setValue(val_si / new_factor)
        row["spin"].blockSignals(False)

        fw = self._fit_widgets.get(param_name)
        if fw:
            lb_si = float(fw["spin_lb"].value()) * old_factor
            ub_si = float(fw["spin_ub"].value()) * old_factor
            fw["spin_lb"].setValue(lb_si / new_factor)
            fw["spin_ub"].setValue(ub_si / new_factor)

        row["unit_index"] = new_index

    # =====================================================================
    # view mode
    # =====================================================================

    def set_view_mode(self, mode: str):
        self.state.view_mode = mode
        self._update_view_buttons()
        self._redraw_all()

    def _on_fit_window_changed(self):
        if self.state.mode in ("simulation", "fitting"):
            self._redraw_all()

    def _fit_window_cycle_indices(self, cycles: int) -> tuple[int, int, int]:
        return self._fit_window_cycle_indices_for(
            self.state.exp_t,
            self.state.exp_R,
            cycles,
            self._get_active_model_key(),
        )

    @staticmethod
    def _fit_window_cycle_indices_for(
        t: np.ndarray | None,
        R: np.ndarray | None,
        cycles: int,
        model_key: str,
    ) -> tuple[int, int, int]:
        if t is None or R is None or t.size == 0 or R.size == 0:
            return 0, 0, 0

        n = min(t.size, R.size)
        R = np.asarray(R[:n], dtype=float)
        finite = np.isfinite(R)
        if not np.any(finite):
            return 0, max(0, n - 1), 0

        valid_idx = np.flatnonzero(finite)
        i_max = int(valid_idx[np.argmax(R[finite])])
        start_idx = i_max if model_key == "NHKV (Rmax)" else 0
        if i_max >= n - 2:
            return start_idx, n - 1, 0

        span = float(np.nanmax(R[finite]) - np.nanmin(R[finite]))
        eps = max(span * 1e-6, np.finfo(float).eps)
        minima: list[int] = []
        i = i_max + 1
        while i < n - 1:
            if not np.isfinite(R[i - 1]) or not np.isfinite(R[i]) or not np.isfinite(R[i + 1]):
                i += 1
                continue

            if R[i] <= R[i - 1] + eps:
                j = i
                while j + 1 < n and np.isfinite(R[j + 1]) and abs(R[j + 1] - R[i]) <= eps:
                    j += 1
                if j + 1 < n and np.isfinite(R[j + 1]) and R[j + 1] > R[j] + eps:
                    minima.append(j)
                    i = j + 1
                    continue
            i += 1

        if not minima:
            return start_idx, n - 1, 0

        found = len(minima)
        end_idx = minima[min(max(1, cycles) - 1, found - 1)]
        return start_idx, end_idx, found

    def _apply_fit_window_cycles(self):
        if self.state.exp_t is None or self.state.exp_R is None:
            return
        cycles = int(self.spin_fit_window_cycles.value())
        if cycles <= 0:
            if self.state.mode in ("simulation", "fitting"):
                self._redraw_all()
            return
        i0, i1, found = self._fit_window_cycle_indices(cycles)
        t = self.state.exp_t
        if t is None or t.size == 0:
            return

        self.spin_t_fit_start.blockSignals(True)
        self.spin_t_fit_end.blockSignals(True)
        self.spin_t_fit_start.setValue(float(t[i0]) * 1e6)
        self.spin_t_fit_end.setValue(float(t[i1]) * 1e6)
        self.spin_t_fit_start.blockSignals(False)
        self.spin_t_fit_end.blockSignals(False)

        if found and found < cycles:
            self.statusBar().showMessage(
                f"Only {found} collapse cycle(s) found; fit window uses the last one found."
            )
        if self.state.mode in ("simulation", "fitting"):
            self._redraw_all()

    def _on_fit_window_auto_changed(self, checked: bool):
        self.spin_t_fit_start.setEnabled(not checked)
        self.spin_t_fit_end.setEnabled(not checked)
        if checked:
            self._apply_fit_window_cycles()
        elif self.state.mode in ("simulation", "fitting"):
            self._redraw_all()

    def _on_fit_window_cycles_changed(self, _value: int):
        if self.chk_fit_window_cycles.isChecked():
            self._apply_fit_window_cycles()

    def _current_fit_window_seconds(
        self,
        t: np.ndarray,
        R: np.ndarray,
        model_key: str,
    ) -> tuple[float, float, int]:
        if self.chk_fit_window_cycles.isChecked():
            cycles = int(self.spin_fit_window_cycles.value())
            if cycles <= 0:
                return float(t[0]), float(t[-1]), cycles
            i0, i1, _found = self._fit_window_cycle_indices_for(t, R, cycles, model_key)
            t_start_s = float(t[i0])
            t_end_s = float(t[i1])
        else:
            t_start_s = float(self.spin_t_fit_start.value()) * 1e-6
            t_end_s = float(self.spin_t_fit_end.value()) * 1e-6
            cycles = int(self.spin_fit_window_cycles.value())
        if t_start_s > t_end_s:
            t_start_s, t_end_s = t_end_s, t_start_s
        return t_start_s, t_end_s, cycles

    def _on_fit_window_dragged(self, which: str, x_view: float):
        if self.state.mode == "jobs":
            return
        if self.chk_fit_window_cycles.isChecked():
            self.chk_fit_window_cycles.setChecked(False)
        t_s = self._time_view_to_s(x_view)
        t_us = t_s * 1e6
        spin = self.spin_t_fit_start if which == "start" else self.spin_t_fit_end
        spin.blockSignals(True)
        spin.setValue(t_us)
        spin.blockSignals(False)

    # =====================================================================
    # zoom callbacks
    # =====================================================================

    def _on_x_zoom_changed(self, value: int):
        self.canvas.zoom_x(value / 100.0)

    def _on_y_zoom_changed(self, value: int):
        self.canvas.zoom_y(value / 100.0)

    def _on_reset_zoom(self):
        self.slider_x_zoom.setValue(100)
        self.slider_y_zoom.setValue(100)
        self.canvas.reset_zoom()

    # =====================================================================
    # redraw
    # =====================================================================

    def _redraw_all(self):
        self.canvas.ax.clear()
        self.canvas.handles = PlotHandles()
        if self.state.view_mode == "dimensional":
            self.canvas.ax.set_xlabel("t (µs)")
            self.canvas.ax.set_ylabel("R (µm)")
        else:
            self.canvas.ax.set_xlabel("t*")
            self.canvas.ax.set_ylabel("R*")
        self.canvas.ax.grid(True, alpha=0.3)

        preview_exp_t = self.state.exp_t
        preview_exp_R = self.state.exp_R
        if self.state.mode == "jobs":
            idx = self._selected_job_index() if hasattr(self, "tbl_jobs") else None
            if idx is not None:
                exp = self._jobs[idx].get("experiment", {})
                preview_exp_t = exp.get("t")
                preview_exp_R = exp.get("R")
                self._job_preview_fit_t = self._jobs[idx].get("best_fit_t")
                self._job_preview_fit_R = self._jobs[idx].get("best_fit_R")
                self._job_preview_fit_meta = self._jobs[idx].get("best_fit_meta")
                self._job_preview_window = self._jobs[idx].get("fit_window")
            else:
                self._job_preview_fit_t = None
                self._job_preview_fit_R = None
                self._job_preview_fit_meta = None
                self._job_preview_window = None

        if preview_exp_t is not None and preview_exp_R is not None:
            t_exp = np.asarray(preview_exp_t, dtype=float).reshape(-1)
            R_exp = np.asarray(preview_exp_R, dtype=float).reshape(-1)
            n_exp = min(t_exp.size, R_exp.size)
            t_exp = t_exp[:n_exp]
            R_exp = R_exp[:n_exp]
        else:
            t_exp = np.array([], dtype=float)
            R_exp = np.array([], dtype=float)

        if self._curve_view_active():
            plotted_curve_count = 0
            for curve in self._view_curves:
                if not curve.get("visible", True):
                    continue
                t_curve = np.asarray(curve.get("t", []), dtype=float).reshape(-1)
                R_curve = np.asarray(curve.get("R", []), dtype=float).reshape(-1)
                n_curve = min(t_curve.size, R_curve.size)
                if n_curve <= 0:
                    continue
                t_curve = t_curve[:n_curve]
                R_curve = R_curve[:n_curve]
                meta = dict(curve.get("meta", {}) or {})
                if self.state.view_mode == "dimensional":
                    t_plot = t_curve * 1e6
                    R_plot = R_curve * 1e6
                else:
                    if curve.get("type") == "simulation" and meta:
                        tc = float(meta.get("tc", 1.0)) or 1.0
                        rmax = float(meta.get("Rmax", 1.0)) or 1.0
                        t_rmax = float(meta.get("t_rmax", 0.0))
                        t_plot = (t_curve - t_rmax) / tc
                        R_plot = R_curve / rmax
                    else:
                        rmax = float(np.nanmax(R_curve)) if R_curve.size else 1.0
                        if not np.isfinite(rmax) or rmax == 0:
                            rmax = 1.0
                        idx_rmax = int(np.nanargmax(R_curve)) if R_curve.size else 0
                        t_rmax = float(t_curve[idx_rmax]) if t_curve.size else 0.0
                        P_inf = float(self.spin_P_inf.value())
                        rho = float(self.spin_rho.value())
                        R_eq = float(self.spin_Req_um.value()) * 1e-6
                        Uc = np.sqrt(P_inf / rho) if rho > 0 else 1.0
                        tc = R_eq / Uc if Uc > 0 else 1.0
                        t_plot = (t_curve - t_rmax) / tc
                        R_plot = R_curve / rmax
                label = str(curve.get("legend", "curve"))
                color = str(curve.get("color", "#1f77b4"))
                width = float(curve.get("width", 1.5))
                if curve.get("type") == "experiment":
                    self.canvas.ax.plot(
                        t_plot,
                        R_plot,
                        linestyle="None",
                        marker="s",
                        markersize=max(1.5, width + 0.5),
                        color=color,
                        label=label,
                    )
                else:
                    self.canvas.ax.plot(
                        t_plot,
                        R_plot,
                        "-",
                        linewidth=width,
                        color=color,
                        label=label,
                    )
                plotted_curve_count += 1
            if 0 < plotted_curve_count <= 16:
                self.canvas.ax.legend(loc="best")
        elif t_exp.size > 0 and R_exp.size > 0:
            if self.state.view_mode == "dimensional":
                t_plot = t_exp * 1e6
                R_plot = R_exp * 1e6
            else:
                P_inf = float(self.spin_P_inf.value())
                rho = float(self.spin_rho.value())
                R_eq = float(self.spin_Req_um.value()) * 1e-6
                Uc = np.sqrt(P_inf / rho) if rho > 0 else 1.0
                tc = R_eq / Uc if Uc > 0 else 1.0
                Rmax = float(np.max(R_exp))
                idx = int(np.argmax(R_exp))
                t_rmax = float(t_exp[idx])
                t_plot = (t_exp - t_rmax) / tc
                R_plot = R_exp / Rmax
            self.canvas.plot_experiment(t_plot, R_plot)
            if t_plot.size > 0:
                self.canvas.set_drag_limits(float(t_plot[0]), float(t_plot[-1]))

        if (
            not self._curve_view_active()
            and
            self.state.sim_t is not None
            and self.state.sim_R is not None
            and self.state.sim_meta is not None
        ):
            t_sim = self.state.sim_t
            R_sim = self.state.sim_R
            meta = self.state.sim_meta
            if self.state.view_mode == "dimensional":
                t_plot = t_sim * 1e6
                R_plot = R_sim * 1e6
            else:
                t_plot = (t_sim - meta["t_rmax"]) / meta["tc"]
                R_plot = R_sim / meta["Rmax"]
            self.canvas.plot_simulation(t_plot, R_plot)

        best_fit_t = self.state.best_fit_t
        best_fit_R = self.state.best_fit_R
        best_fit_meta = self.state.best_fit_meta
        if self.state.mode == "jobs":
            best_fit_t = getattr(self, "_job_preview_fit_t", None)
            best_fit_R = getattr(self, "_job_preview_fit_R", None)
            best_fit_meta = getattr(self, "_job_preview_fit_meta", None)

        if best_fit_t is not None and best_fit_R is not None:
            bf_meta = best_fit_meta
            if self.state.view_mode == "dimensional":
                t_plot = best_fit_t * 1e6
                R_plot = best_fit_R * 1e6
            else:
                if bf_meta:
                    t_plot = (
                        (best_fit_t - bf_meta.get("t_rmax", 0))
                        / bf_meta.get("tc", 1)
                    )
                    R_plot = best_fit_R / bf_meta.get("Rmax", 1)
                else:
                    t_plot = best_fit_t
                    R_plot = best_fit_R
            self.canvas.plot_fit_best(t_plot, R_plot)

        hide_fit_window = (
            self.state.mode == "simulation"
            and self.chk_fit_window_cycles.isChecked()
            and int(self.spin_fit_window_cycles.value()) <= 0
        )
        if (
            not hide_fit_window
            and self.state.mode in ("simulation", "fitting")
            and self.state.exp_t is not None
        ):
            t0_view = self._time_s_to_view(
                float(self.spin_t_fit_start.value()) * 1e-6
            )
            t1_view = self._time_s_to_view(
                float(self.spin_t_fit_end.value()) * 1e-6
            )
            self.canvas.draw_fit_window(min(t0_view, t1_view), max(t0_view, t1_view))
        elif self.state.mode == "jobs":
            fit_window = getattr(self, "_job_preview_window", None)
            if fit_window:
                t0_view = self._time_s_to_view(float(fit_window.get("t_start_s", 0.0)))
                t1_view = self._time_s_to_view(float(fit_window.get("t_end_s", 0.0)))
                self.canvas.draw_fit_window(min(t0_view, t1_view), max(t0_view, t1_view))

        self.canvas.draw_idle()

        # Store data bounds for zoom, then apply current slider zoom
        has_data = (
            self.canvas.handles.exp_line is not None
            or self.canvas.handles.sim_line is not None
            or self.canvas.handles.fit_best_line is not None
            or bool(self.canvas.ax.lines)
            or bool(self.canvas.ax.collections)
        )
        if has_data:
            xlim = self.canvas.ax.get_xlim()
            ylim = self.canvas.ax.get_ylim()
            self.canvas.set_data_bounds(xlim, ylim)
            x_frac = self.slider_x_zoom.value() / 100.0
            y_frac = self.slider_y_zoom.value() / 100.0
            if x_frac < 0.999:
                self.canvas.zoom_x(x_frac)
            if y_frac < 0.999:
                self.canvas.zoom_y(y_frac)

    # =====================================================================
    # import wizard
    # =====================================================================

    @staticmethod
    def _wizard_load_mat(path: str) -> dict:
        try:
            return loadmat(path, squeeze_me=True, struct_as_record=False)
        except Exception:
            if _HAS_MAT73:
                return _mat73.loadmat(path)
            raise

    @staticmethod
    def _wizard_flatten_namespace(mat: dict) -> dict:
        flat: dict[str, object] = {}

        def add(prefix: str, value, depth: int = 0):
            if not prefix or prefix.startswith("__"):
                return
            flat[prefix] = value
            if depth >= 2:
                return
            if isinstance(value, dict):
                for key, sub in value.items():
                    add(f"{prefix}.{key}", sub, depth + 1)
                return
            if hasattr(value, "_fieldnames"):
                for field in value._fieldnames:
                    add(f"{prefix}.{field}", getattr(value, field), depth + 1)
                return
            arr = np.asarray(value)
            if arr.dtype.kind == "O" and arr.size == 1:
                try:
                    inner = arr.reshape(-1)[0]
                    if hasattr(inner, "_fieldnames"):
                        for field in inner._fieldnames:
                            add(f"{prefix}.{field}", getattr(inner, field), depth + 1)
                    elif isinstance(inner, dict):
                        for key, sub in inner.items():
                            add(f"{prefix}.{key}", sub, depth + 1)
                except Exception:
                    pass

        for key, value in mat.items():
            add(str(key), value)
        return flat

    @staticmethod
    def _wizard_is_numeric_array(value) -> bool:
        try:
            arr = np.asarray(value)
            if arr.dtype.kind == "O":
                return False
            arr = np.squeeze(arr)
            return arr.ndim == 1 and arr.size > 0 and arr.dtype.kind in "biufc"
        except Exception:
            return False

    @staticmethod
    def _wizard_is_scalar_like(value) -> bool:
        try:
            arr = np.asarray(value)
            if arr.dtype.kind == "O":
                return False
            arr = np.squeeze(arr)
            return arr.size == 1 and arr.dtype.kind in "biufc"
        except Exception:
            return False

    @staticmethod
    def _wizard_to_1d_float(value) -> np.ndarray:
        arr = np.asarray(value)
        arr = np.squeeze(arr)
        if arr.ndim != 1:
            raise ValueError(f"Expected a 1-D numeric array, got shape {arr.shape}.")
        return arr.astype(float)

    @staticmethod
    def _wizard_to_float(value, default=None):
        try:
            arr = np.asarray(value)
            if arr.size == 0:
                return default
            return float(arr.astype(float).reshape(-1)[0])
        except Exception:
            return default

    def _wizard_to_string(self, value, default: str = "") -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        try:
            arr = np.asarray(value)
            if arr.size == 0:
                return default
            if arr.dtype.kind in ("U", "S"):
                parts = [
                    x.decode(errors="replace") if isinstance(x, bytes) else str(x)
                    for x in arr.reshape(-1)
                ]
                if len(parts) > 1 and all(len(part) <= 1 for part in parts):
                    return "".join(parts)
                return parts[0]
            first = arr.reshape(-1)[0]
            if isinstance(first, bytes):
                return first.decode(errors="replace")
            return str(first)
        except Exception:
            return default

    @staticmethod
    def _wizard_guess_key(keys: list[str], candidates: list[str]) -> str | None:
        lower = {key.lower(): key for key in keys}
        for cand in candidates:
            hit = lower.get(cand.lower())
            if hit is not None:
                return hit
        for cand in candidates:
            cand_l = cand.lower()
            for key in keys:
                name = key.lower().split(".")[-1]
                if name == cand_l or name.startswith(cand_l):
                    return key
        return None

    def _wizard_combo(
        self,
        keys: list[str],
        candidates: list[str],
        scalar_only: bool = False,
        flat: dict | None = None,
    ) -> QComboBox:
        combo = _NoWheelComboBox()
        combo.addItem("(none)", None)
        for key in keys:
            if scalar_only and flat is not None and not self._wizard_is_scalar_like(flat[key]):
                continue
            combo.addItem(key, key)
        guess_keys = [combo.itemText(i) for i in range(1, combo.count())]
        guess = self._wizard_guess_key(guess_keys, candidates)
        if guess:
            idx = combo.findText(guess)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        return combo

    def _import_unit_to_si(self, name: str, value, unit: str):
        if value is None:
            return None
        if unit == "us":
            return np.asarray(value, dtype=float) * 1e-6
        if unit == "um":
            return np.asarray(value, dtype=float) * 1e-6
        if unit == "pixel":
            return np.asarray(value, dtype=float) * float(self._import_wizard_um_per_pixel) * 1e-6
        return value

    def _confirm_import_default_units(
        self,
        *,
        parent,
        used_fps: bool,
        r_unit: str,
        t_unit: str,
        accepted_state: dict | None = None,
    ) -> bool:
        uses_conversion = used_fps or r_unit in ("um", "pixel") or t_unit == "us"
        risky_pixel_with_time = (not used_fps) and r_unit == "pixel"
        if not uses_conversion and not risky_pixel_with_time:
            return True
        if accepted_state is not None and accepted_state.get("accepted", False):
            return True
        parts = []
        if used_fps:
            parts.append(f"t_exp is reconstructed from fps={float(self._import_wizard_fps):.3g}.")
        elif t_unit == "us":
            parts.append("t_exp is converted from us to seconds.")
        if r_unit == "um":
            parts.append("R_exp is converted from um to meters.")
        elif r_unit == "pixel":
            parts.append(
                f"R_exp is converted from pixels using {float(self._import_wizard_um_per_pixel):.3g} um/pixel."
            )
        if risky_pixel_with_time:
            parts.append(
                "A time-axis variable is present while R_exp is set to pixel; "
                "choose m or um instead if R is already physical radius."
            )
        reply = QMessageBox.warning(
            parent or self,
            "Confirm import units",
            "\n".join(parts) + "\n\nContinue importing with these settings?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        ok = reply == QMessageBox.StandardButton.Yes
        if ok and accepted_state is not None:
            accepted_state["accepted"] = True
        return ok

    def _load_experiment_with_import_defaults(
        self,
        path: str,
        *,
        confirm_units: bool = False,
        parent=None,
        accepted_state: dict | None = None,
    ) -> ExperimentData:
        mat = self._wizard_load_mat(path)
        flat = {
            key: value for key, value in self._wizard_flatten_namespace(mat).items()
            if not str(key).startswith("__")
        }
        keys = sorted(flat.keys(), key=str.lower)
        array_keys = [key for key in keys if self._wizard_is_numeric_array(flat[key])]
        scalar_keys = [key for key in keys if self._wizard_is_scalar_like(flat[key])]
        t_key = self._wizard_guess_key(array_keys, self._import_wizard_keywords["t_exp"])
        R_key = self._wizard_guess_key(array_keys, self._import_wizard_keywords["R_exp"])
        if not R_key:
            raise ValueError("Could not find R_exp using Import Wizard recognition names.")

        r_unit = self._import_wizard_units.get("R_exp", "m")
        t_unit = self._import_wizard_units.get("t_exp", "s")
        R = self._wizard_to_1d_float(flat[R_key])
        R = np.asarray(self._import_unit_to_si("R_exp", R, r_unit), dtype=float)
        used_fps = False
        if t_key:
            t = self._wizard_to_1d_float(flat[t_key])
            t = np.asarray(self._import_unit_to_si("t_exp", t, t_unit), dtype=float)
        else:
            fps = float(self._import_wizard_fps)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("fps must be positive to reconstruct t_exp.")
            t = np.arange(R.shape[0], dtype=float) / fps
            used_fps = True
        if t.shape[0] != R.shape[0]:
            raise ValueError(f"t_exp and R_exp length mismatch: {t.shape[0]} vs {R.shape[0]}.")
        if confirm_units and not self._confirm_import_default_units(
            parent=parent or self,
            used_fps=used_fps,
            r_unit=r_unit,
            t_unit=t_unit,
            accepted_state=accepted_state,
        ):
            raise RuntimeError("Import cancelled by user.")

        finite = np.isfinite(t) & np.isfinite(R)
        if int(np.count_nonzero(finite)) < 3:
            raise ValueError("Experiment data must contain at least 3 finite t/R pairs.")
        t = t[finite]
        R = R[finite]
        n_before_cleanup = int(R.size)
        cleanup_mask = np.ones(R.shape[0], dtype=bool)
        if self._import_wizard_remove_below:
            cleanup_mask &= R >= 0.0
        if self._import_wizard_remove_spikes and R.size >= 3:
            kernel = min(5, R.size if R.size % 2 == 1 else R.size - 1)
            if kernel >= 3:
                R_med = medfilt(R.astype(float), kernel_size=kernel)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = np.where(R_med > 0, R / R_med, 1.0)
                cleanup_mask &= ratio <= float(self._import_wizard_spike_threshold)
        if not np.all(cleanup_mask):
            t = t[cleanup_mask]
            R = R[cleanup_mask]
        if R.size < 3:
            raise ValueError(
                f"Experiment data has fewer than 3 points after cleanup ({R.size})."
            )
        n_after_cleanup = int(R.size)
        order = np.argsort(t)
        t = t[order]
        R = R[order]
        t = t - _find_rmax_time(t, R, n_pts=7)

        def scalar(name: str):
            key = self._wizard_guess_key(scalar_keys, self._import_wizard_keywords[name])
            if not key:
                return None
            val = self._wizard_to_float(flat[key], None)
            if val is None:
                return None
            unit = self._import_wizard_units.get(name, "")
            converted = self._import_unit_to_si(name, val, unit)
            try:
                return float(np.asarray(converted).reshape(-1)[0])
            except Exception:
                return None

        exp = ExperimentData(
            t=t,
            R=R,
            source_path=path,
            t_key=t_key or f"fps:{float(self._import_wizard_fps):.6g}",
            R_key=R_key,
            P_inf=scalar("P_inf"),
            rho=scalar("rho"),
            R_eq=scalar("Req"),
        )
        exp_meta = {
            "method": "import_defaults",
            "source_path": path,
            "t_key": t_key or "",
            "R_key": R_key,
            "t_unit": t_unit,
            "R_unit": r_unit,
            "used_fps": bool(used_fps),
            "fps": float(self._import_wizard_fps),
            "um_per_pixel": float(self._import_wizard_um_per_pixel),
            "remove_negative_R": bool(self._import_wizard_remove_below),
            "remove_isolated_spikes": bool(self._import_wizard_remove_spikes),
            "spike_threshold": float(self._import_wizard_spike_threshold),
            "n_points_before_cleanup": n_before_cleanup,
            "n_points_after_cleanup": n_after_cleanup,
            "n_points_removed_cleanup": n_before_cleanup - n_after_cleanup,
        }
        object.__setattr__(exp, "import_metadata", exp_meta)
        return exp

    def _apply_import_wizard_experiment(
        self,
        *,
        path: str,
        t: np.ndarray,
        R: np.ndarray,
        legend: str,
        Req: float | None,
        P_inf: float | None,
        rho: float | None,
        c_long: float | None,
        gamma: float | None,
        t_sim: np.ndarray | None = None,
        R_sim: np.ndarray | None = None,
        import_metadata: dict | None = None,
    ):
        finite = np.isfinite(t) & np.isfinite(R)
        if int(np.count_nonzero(finite)) < 3:
            raise ValueError("Experiment data must contain at least 3 finite t/R pairs.")
        t = np.asarray(t[finite], dtype=float).reshape(-1)
        R = np.asarray(R[finite], dtype=float).reshape(-1)
        order = np.argsort(t)
        t = t[order]
        R = R[order]
        t = t - _find_rmax_time(t, R, n_pts=7)

        sim_t_clean = None
        sim_R_clean = None
        sim_meta = None
        if t_sim is not None or R_sim is not None:
            if t_sim is None or R_sim is None:
                raise ValueError("Please map both t_sim and R_sim, or leave both empty.")
            t_sim_arr = np.asarray(t_sim, dtype=float).reshape(-1)
            R_sim_arr = np.asarray(R_sim, dtype=float).reshape(-1)
            if t_sim_arr.shape[0] != R_sim_arr.shape[0]:
                raise ValueError(
                    f"t_sim and R_sim length mismatch: {t_sim_arr.shape[0]} vs {R_sim_arr.shape[0]}."
                )
            finite_sim = np.isfinite(t_sim_arr) & np.isfinite(R_sim_arr)
            if int(np.count_nonzero(finite_sim)) >= 2:
                sim_t_clean = t_sim_arr[finite_sim]
                sim_R_clean = R_sim_arr[finite_sim]
                sim_order = np.argsort(sim_t_clean)
                sim_t_clean = sim_t_clean[sim_order]
                sim_R_clean = sim_R_clean[sim_order]
                rmax = float(np.nanmax(sim_R_clean)) if sim_R_clean.size else 1.0
                idx = int(np.nanargmax(sim_R_clean)) if sim_R_clean.size else 0
                t_rmax = float(sim_t_clean[idx]) if sim_t_clean.size else 0.0
                p_val = float(P_inf) if P_inf is not None else float(self.spin_P_inf.value())
                rho_val = float(rho) if rho is not None else float(self.spin_rho.value())
                uc = math.sqrt(p_val / rho_val) if p_val > 0.0 and rho_val > 0.0 else 1.0
                tc = rmax / uc if uc > 0.0 and rmax > 0.0 else 1.0
                sim_meta = {"Rmax": rmax, "t_rmax": t_rmax, "tc": tc}

        if self._curve_view_active():
            self._add_curve_to_view(
                curve_type=self._infer_curve_type(t, R),
                t=t,
                R=R,
                legend=legend or Path(path).stem,
            )
            if sim_t_clean is not None and sim_R_clean is not None:
                self._add_curve_to_view(
                    curve_type="simulation",
                    t=sim_t_clean,
                    R=sim_R_clean,
                    legend=f"{legend or Path(path).stem} simulation",
                    meta=sim_meta,
                )
            self.statusBar().showMessage(f"Imported curve(s) into Curve View: {Path(path).name}")
            self._redraw_all()
            return

        if self.state.mode == "jobs":
            raise ValueError("Switch to Simulation or Fitting mode before importing experiment data.")

        self.state.exp_t = t
        self.state.exp_R = R
        self.state.exp_path = path
        self.state.import_metadata = dict(import_metadata or {})
        self.state.P_inf = P_inf
        self.state.rho = rho
        self.state.R_eq = Req
        self._editing_job_index = None
        if self.state.mode == "fitting":
            self.btn_add_job.setText("Add to job list")
            self.btn_add_job.setToolTip("Add the current fitting setup as a queued job.")

        self.state.sim_t = sim_t_clean
        self.state.sim_R = sim_R_clean
        self.state.sim_meta = sim_meta
        self.state.best_fit_t = None
        self.state.best_fit_R = None
        self.state.best_fit_meta = None

        if Req is None and R.size > 0:
            self.state.R_eq = float(np.mean(R[-min(20, R.size):]))
        if self.state.R_eq is not None:
            self.spin_Req_um.setValue(float(self.state.R_eq) * 1e6)
        if P_inf is not None:
            self.spin_P_inf.setValue(float(P_inf))
        if rho is not None:
            self.spin_rho.setValue(float(rho))
        if c_long is not None:
            self.spin_c_long.setValue(float(c_long))
        if gamma is not None:
            self.spin_gamma.setValue(float(gamma))

        if t.size > 0:
            self.spin_t_fit_start.setValue(float(t[0]) * 1e6)
            self.spin_t_fit_end.setValue(float(t[-1]) * 1e6)
            if self.chk_fit_window_cycles.isChecked():
                self._apply_fit_window_cycles()

        self._update_window_title()
        self.statusBar().showMessage(f"Imported experiment with Import Wizard: {Path(path).name}")
        self._redraw_all()

    def on_import_wizard(self, path: str | None = None):
        if not isinstance(path, str):
            path = None
        dlg = QDialog(self)
        dlg.setWindowTitle("Import Wizard")
        dlg.setMinimumSize(820, 620)
        dlg.setModal(False)

        root = QVBoxLayout(dlg)

        top = QHBoxLayout()
        left_buttons = QVBoxLayout()
        btn_import = QPushButton("Import")
        btn_format = QPushButton("Variable format")
        btn_format.setCursor(Qt.CursorShape.WhatsThisCursor)
        btn_format.setStyleSheet(
            "QPushButton {"
            "border: 1px solid #666;"
            "border-radius: 9px;"
            "padding: 2px 8px;"
            "background: transparent;"
            "color: palette(window-text);"
            "text-align: left;"
            "}"
            "QPushButton:hover { background: rgba(255, 255, 255, 0.06); }"
        )
        btn_format.setToolTip(
            "Recommended MAT variables:\n"
            "- t or t_exp: experimental time in seconds\n"
            "- R or R_exp: experimental radius in meters\n"
            "- t_sim / R_sim: simulation curve, if present\n"
            "- Req, R_eq, or R_equilibrium: equilibrium radius in meters\n"
            "- legend: curve label string\n"
            "- P_inf, rho, c_long, gamma: physical constants in SI units"
        )
        btn_units = QPushButton("Use um/us")
        left_buttons.addWidget(btn_import)
        left_buttons.addWidget(btn_format)
        left_buttons.addWidget(btn_units)
        left_buttons.addStretch(1)
        top.addLayout(left_buttons)
        drop = _MatDropFrame()
        drop.setMinimumHeight(92)
        top.addWidget(drop, stretch=1)
        root.addLayout(top)

        lbl_path = QLabel("No MAT file loaded.")
        lbl_path.setWordWrap(True)
        root.addWidget(lbl_path)

        grp_calibration = QGroupBox("Calibration")
        cal_layout = QHBoxLayout(grp_calibration)
        spin_um_per_pixel = _SigFigSpinBox()
        spin_um_per_pixel.setRange(1e-6, 1e6)
        spin_um_per_pixel.setDecimals(6)
        spin_um_per_pixel.setValue(float(self._import_wizard_um_per_pixel))
        spin_um_per_pixel.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin_um_per_pixel.setToolTip("Radius conversion when R_exp unit is pixel.")
        spin_fps = _SigFigSpinBox()
        spin_fps.setRange(1.0, 1e12)
        spin_fps.setDecimals(3)
        spin_fps.setValue(float(self._import_wizard_fps))
        spin_fps.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin_fps.setToolTip("Frame rate used to reconstruct t_exp when no time variable is mapped.")
        lbl_um_per_pixel = QLabel("um/pixel:")
        lbl_fps = QLabel("fps:")
        cal_layout.addWidget(lbl_um_per_pixel)
        cal_layout.addWidget(spin_um_per_pixel, stretch=1)
        cal_layout.addSpacing(16)
        cal_layout.addWidget(lbl_fps)
        cal_layout.addWidget(spin_fps, stretch=1)
        root.addWidget(grp_calibration)
        lbl_converted_preview = QLabel("Converted preview: select R_exp to preview")
        lbl_converted_preview.setWordWrap(True)
        root.addWidget(lbl_converted_preview)
        spin_um_per_pixel.valueChanged.connect(lambda _v: update_converted_preview())
        spin_fps.valueChanged.connect(lambda _v: update_converted_preview())

        body = QHBoxLayout()
        grp_arrays = QGroupBox("Curve arrays")
        arr_form = QFormLayout(grp_arrays)
        grp_scalars = QGroupBox("Metadata and physical constants")
        scalar_form = QFormLayout(grp_scalars)
        body.addWidget(grp_arrays, stretch=1)
        body.addWidget(grp_scalars, stretch=1)
        root.addLayout(body, stretch=1)

        grp_cleanup = QGroupBox("Data cleanup")
        cleanup_layout = QHBoxLayout(grp_cleanup)
        chk_remove_below = QCheckBox("Remove negative R")
        chk_remove_below.setChecked(bool(self._import_wizard_remove_below))
        chk_remove_spikes = QCheckBox("Remove isolated spikes")
        chk_remove_spikes.setChecked(bool(self._import_wizard_remove_spikes))
        chk_remove_below.setToolTip("Remove radii below 0 after unit conversion.")
        spin_spike_threshold = _SigFigSpinBox()
        spin_spike_threshold.setRange(1.0, 1e6)
        spin_spike_threshold.setDecimals(6)
        spin_spike_threshold.setValue(float(self._import_wizard_spike_threshold))
        spin_spike_threshold.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin_spike_threshold.setToolTip("Spike ratio threshold relative to local median.")
        cleanup_layout.addWidget(chk_remove_below)
        cleanup_layout.addWidget(chk_remove_spikes)
        cleanup_layout.addSpacing(12)
        cleanup_layout.addWidget(QLabel("spike threshold:"))
        cleanup_layout.addWidget(spin_spike_threshold, stretch=1)
        root.addWidget(grp_cleanup)

        btn_apply = QPushButton("Import data")
        btn_apply.setEnabled(False)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        btn_set_default = QPushButton("Learn names")
        btn_set_default.setToolTip(
            "Add the currently selected MAT variable names to the Import Wizard "
            "recognition list.\n"
            "Existing names are kept; selected names are tried first next time.\n"
            "Also saves current calibration and cleanup settings."
        )
        btn_reset = QPushButton("Reset names")
        btn_reset.setToolTip(
            "Reset Import Wizard variable-name recognition to the built-in defaults."
        )
        buttons.addWidget(btn_set_default)
        buttons.addWidget(btn_reset)
        buttons.addWidget(btn_apply)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.close)
        buttons.addWidget(btn_close)
        root.addLayout(buttons)

        ctx = {"path": "", "flat": {}, "pixel_time_warning_accepted": False}
        array_rows: dict[str, QComboBox] = {}
        scalar_rows: dict[str, tuple[QComboBox, QLineEdit]] = {}
        unit_rows: dict[str, QComboBox] = {}
        using_micro_units = {"value": False}

        def clear_layout(layout: QFormLayout):
            while layout.rowCount():
                layout.removeRow(0)

        def unit_options_for_name(name: str) -> list[str]:
            if name in ("t_exp", "t_sim", "t_start", "t_end"):
                return ["s", "us"]
            if name == "R_exp":
                return ["m", "um", "pixel"]
            if name in ("R_sim", "Req", "Rmax"):
                return ["m", "um"]
            fixed = {
                "legend": "",
                "P_inf": "Pa",
                "rho": "kg/m^3",
                "c_long": "m/s",
                "gamma": "N/m",
                "LSQErr": "um^2/point",
            }
            if name in fixed:
                return [fixed[name]]
            model = self._get_active_model()
            for param in getattr(model, "parameters", []):
                if param.name == name and param.units:
                    return [str(param.units[0].label)]
            return [""]

        def unit_combo_for_name(name: str) -> QComboBox:
            combo = _NoWheelComboBox()
            for unit in unit_options_for_name(name):
                combo.addItem(unit, unit)
            defaults = {
                "t_exp": "s",
                "t_sim": "s",
                "t_start": "s",
                "t_end": "s",
                "R_exp": "m",
                "R_sim": "m",
                "Req": "m",
                "Rmax": "m",
            }
            default = self._import_wizard_units.get(name, defaults.get(name))
            if default:
                idx = combo.findData(default)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.setMinimumWidth(58)
            unit_rows[name] = combo
            return combo

        def unit_label_for_name(name: str) -> QLabel:
            opts = unit_options_for_name(name)
            label = QLabel(opts[0] if opts else "")
            label.setMinimumWidth(64)
            return label

        def unit_for_preview(name: str) -> str:
            combo = unit_rows.get(name)
            if combo is None:
                opts = unit_options_for_name(name)
                return opts[0] if opts else ""
            return str(combo.currentData() or "")

        def to_si(name: str, value):
            if value is None:
                return None
            unit = unit_for_preview(name)
            if unit == "us":
                return np.asarray(value, dtype=float) * 1e-6
            if unit == "um":
                return np.asarray(value, dtype=float) * 1e-6
            if unit == "pixel":
                return np.asarray(value, dtype=float) * float(spin_um_per_pixel.value()) * 1e-6
            return value

        def display_scalar(name: str, value: float) -> float:
            unit = unit_for_preview(name)
            if unit == "um":
                return float(value) * 1e6
            if unit == "us":
                return float(value) * 1e6
            return float(value)

        def sci3(value: float, unit: str = "") -> str:
            text = f"{float(value):.2e}"
            return f"{text} {unit}".rstrip()

        def preview_number(name: str, value: float) -> str:
            if name in ("Req", "Rmax") and unit_for_preview(name) == "um":
                return f"{float(value):.3g}"
            return sci3(value)

        def scalar_preview(name: str, key: str | None) -> str:
            defaults = {
                "Req": "",
                "legend": Path(ctx["path"]).stem if ctx["path"] else "",
                "P_inf": sci3(float(self.spin_P_inf.value())),
                "rho": sci3(float(self.spin_rho.value())),
                "c_long": sci3(float(self.spin_c_long.value())),
                "gamma": sci3(float(self.spin_gamma.value())),
                "Rmax": "",
                "t_start": "",
                "t_end": "",
                "LSQErr": "",
            }
            if not key:
                return defaults.get(name, "")
            value = ctx["flat"].get(key)
            if name == "legend":
                return self._normalise_legend_text(self._wizard_to_string(value, defaults["legend"]))
            val = self._wizard_to_float(value, None)
            return "" if val is None else preview_number(name, display_scalar(name, val))

        def make_scalar_preview_edit() -> QLineEdit:
            preview = QLineEdit()
            preview.setToolTip("Editable preview value. Manual edits are used when importing.")
            preview.setProperty("manual_edited", False)
            preview.textEdited.connect(lambda _text, pv=preview: pv.setProperty("manual_edited", True))
            return preview

        def set_scalar_preview_text(name: str, combo: QComboBox, preview: QLineEdit):
            preview.setText(scalar_preview(name, combo.currentData()))
            if name == "Req" and not combo.currentData():
                preview.setPlaceholderText("auto from R")
            else:
                preview.setPlaceholderText("")
            preview.setProperty("manual_edited", False)

        def update_converted_preview():
            try:
                r_combo = array_rows.get("R_exp")
                if r_combo is None or not r_combo.currentData():
                    lbl_converted_preview.setText("Converted preview: select R_exp to preview")
                    return
                R_raw = self._wizard_to_1d_float(ctx["flat"][r_combo.currentData()])
                R_si = np.asarray(to_si("R_exp", R_raw), dtype=float)
                finite_R = R_si[np.isfinite(R_si)]
                if finite_R.size == 0:
                    lbl_converted_preview.setText("Converted preview: no finite R_exp values")
                    return
                t_combo = array_rows.get("t_exp")
                if t_combo is not None and t_combo.currentData():
                    t_raw = self._wizard_to_1d_float(ctx["flat"][t_combo.currentData()])
                    t_si = np.asarray(to_si("t_exp", t_raw), dtype=float)
                    t_source = str(t_combo.currentData())
                else:
                    fps = float(spin_fps.value())
                    t_si = np.arange(R_si.size, dtype=float) / fps if fps > 0 else np.array([], dtype=float)
                    t_source = "fps"
                n0 = min(R_si.size, t_si.size)
                if n0 > 0 and (chk_remove_below.isChecked() or chk_remove_spikes.isChecked()):
                    mask = np.ones(n0, dtype=bool)
                    R_for_mask = R_si[:n0]
                    if chk_remove_below.isChecked():
                        mask &= R_for_mask >= 0.0
                    if chk_remove_spikes.isChecked() and n0 >= 3:
                        kernel = min(5, n0 if n0 % 2 == 1 else n0 - 1)
                        if kernel >= 3:
                            R_med = medfilt(R_for_mask.astype(float), kernel_size=kernel)
                            with np.errstate(invalid="ignore", divide="ignore"):
                                ratio = np.where(R_med > 0, R_for_mask / R_med, 1.0)
                            mask &= ratio <= float(spin_spike_threshold.value())
                    R_si = R_si[:n0][mask]
                    t_si = t_si[:n0][mask]
                R_preview = R_si[:min(R_si.size, t_si.size)]
                finite_R = R_preview[np.isfinite(R_preview)]
                n = min(R_si.size, t_si.size)
                if n <= 0:
                    duration_us = float("nan")
                else:
                    t_finite = t_si[:n][np.isfinite(t_si[:n])]
                    duration_us = (
                        float(np.nanmax(t_finite) - np.nanmin(t_finite)) * 1e6
                        if t_finite.size
                        else float("nan")
                    )
                rmax_um = float(np.nanmax(finite_R)) * 1e6
                req_um = float(np.mean(finite_R[-min(20, finite_R.size):])) * 1e6
                lbl_converted_preview.setText(
                    f"Converted preview: n = {n} | Rmax = {rmax_um:.3g} um | "
                    f"Req = {req_um:.3g} um | "
                    f"duration = {duration_us:.3g} us | t source = {t_source}"
                )
            except Exception as exc:
                lbl_converted_preview.setText(f"Converted preview: {exc}")

        chk_remove_below.stateChanged.connect(lambda _v: update_converted_preview())
        chk_remove_spikes.stateChanged.connect(lambda _v: update_converted_preview())
        spin_spike_threshold.valueChanged.connect(lambda _v: update_converted_preview())

        def rebuild_for_path(path: str):
            mat = self._wizard_load_mat(path)
            flat = {
                key: value for key, value in self._wizard_flatten_namespace(mat).items()
                if not str(key).startswith("__")
            }
            keys = sorted(flat.keys(), key=str.lower)
            ctx["path"] = path
            ctx["flat"] = flat
            lbl_path.setText(path)
            clear_layout(arr_form)
            clear_layout(scalar_form)
            array_rows.clear()
            scalar_rows.clear()
            unit_rows.clear()

            array_specs = [
                ("t_exp", self._import_wizard_keywords["t_exp"]),
                ("R_exp", self._import_wizard_keywords["R_exp"]),
                ("t_sim", self._import_wizard_keywords["t_sim"]),
                ("R_sim", self._import_wizard_keywords["R_sim"]),
                ("legend", self._import_wizard_keywords["legend"]),
                ("Req", self._import_wizard_keywords["Req"]),
                ("Rmax", self._import_wizard_keywords["Rmax"]),
            ]
            array_keys = [key for key in keys if self._wizard_is_numeric_array(flat[key])]
            for i, (label, candidates) in enumerate(array_specs):
                if i == 4:
                    line = QFrame()
                    line.setFrameShape(QFrame.Shape.HLine)
                    line.setFrameShadow(QFrame.Shadow.Sunken)
                    arr_form.addRow(line)
                if label == "legend":
                    combo = self._wizard_combo(keys, candidates, scalar_only=False, flat=flat)
                    preview = make_scalar_preview_edit()
                    set_scalar_preview_text(label, combo, preview)
                    combo.currentIndexChanged.connect(
                        functools.partial(
                            lambda _idx, nm, cb, pv: (set_scalar_preview_text(nm, cb, pv), update_converted_preview()),
                            nm=label,
                            cb=combo,
                            pv=preview,
                        )
                    )
                    scalar_rows[label] = (combo, preview)
                    row = QHBoxLayout()
                    row.addWidget(combo, stretch=9)
                    row.addWidget(preview, stretch=8)
                    arr_form.addRow(f"{label}:", row)
                elif label in ("Req", "Rmax"):
                    combo = self._wizard_combo(keys, candidates, scalar_only=True, flat=flat)
                    preview = make_scalar_preview_edit()
                    set_scalar_preview_text(label, combo, preview)
                    combo.currentIndexChanged.connect(
                        functools.partial(
                            lambda _idx, nm, cb, pv: (set_scalar_preview_text(nm, cb, pv), update_converted_preview()),
                            nm=label,
                            cb=combo,
                            pv=preview,
                        )
                    )
                    scalar_rows[label] = (combo, preview)
                    row = QHBoxLayout()
                    row.addWidget(combo, stretch=2)
                    unit = unit_combo_for_name(label)
                    row.addWidget(unit)
                    row.addWidget(preview, stretch=1)
                    unit.currentIndexChanged.connect(
                        functools.partial(
                            lambda _idx, nm, cb, pv: (set_scalar_preview_text(nm, cb, pv), update_converted_preview()),
                            nm=label,
                            cb=combo,
                            pv=preview,
                        )
                    )
                    arr_form.addRow(f"{label}:", row)
                else:
                    combo = self._wizard_combo(array_keys, candidates)
                    array_rows[label] = combo
                    combo.currentIndexChanged.connect(lambda _idx: update_converted_preview())
                    row = QHBoxLayout()
                    row.addWidget(combo, stretch=1)
                    unit = unit_combo_for_name(label)
                    row.addWidget(unit)
                    unit.currentIndexChanged.connect(lambda _idx: update_converted_preview())
                    arr_form.addRow(f"{label}:", row)

            def sync_fps_enabled():
                enabled = array_rows.get("t_exp") is not None and not array_rows["t_exp"].currentData()
                spin_fps.setEnabled(bool(enabled))
                lbl_fps.setEnabled(bool(enabled))

            if "t_exp" in array_rows:
                array_rows["t_exp"].currentIndexChanged.connect(lambda _idx: (sync_fps_enabled(), update_converted_preview()))
            sync_fps_enabled()

            scalar_specs = [
                ("P_inf", self._import_wizard_keywords["P_inf"]),
                ("rho", self._import_wizard_keywords["rho"]),
                ("c_long", self._import_wizard_keywords["c_long"]),
                ("gamma", self._import_wizard_keywords["gamma"]),
                ("t_start", self._import_wizard_keywords["t_start"]),
                ("t_end", self._import_wizard_keywords["t_end"]),
                ("LSQErr", self._import_wizard_keywords["LSQErr"]),
            ]
            for label, candidates in scalar_specs:
                combo = self._wizard_combo(keys, candidates, scalar_only=(label != "legend"), flat=flat)
                preview = make_scalar_preview_edit()
                set_scalar_preview_text(label, combo, preview)
                combo.currentIndexChanged.connect(
                    functools.partial(
                        lambda _idx, nm, cb, pv: (set_scalar_preview_text(nm, cb, pv), update_converted_preview()),
                        nm=label,
                        cb=combo,
                        pv=preview,
                    )
                )
                scalar_rows[label] = (combo, preview)
                row = QHBoxLayout()
                row.addWidget(combo, stretch=2)
                row.addWidget(preview, stretch=1)
                row.addWidget(unit_label_for_name(label))
                scalar_form.addRow(f"{label}:", row)

            btn_apply.setEnabled(True)
            update_converted_preview()

        def choose_file():
            path, _ = QFileDialog.getOpenFileName(dlg, "Import MAT file", "", "MAT files (*.mat)")
            if path:
                try:
                    rebuild_for_path(path)
                except Exception as exc:
                    QMessageBox.critical(dlg, "Import failed", f"{exc}\n\n{traceback.format_exc()}")

        def dropped_file(path: str):
            try:
                rebuild_for_path(path)
            except Exception as exc:
                QMessageBox.critical(dlg, "Import failed", f"{exc}\n\n{traceback.format_exc()}")

        def selected_array(name: str) -> np.ndarray | None:
            key = array_rows[name].currentData()
            if not key:
                return None
            arr = self._wizard_to_1d_float(ctx["flat"][key])
            return np.asarray(to_si(name, arr), dtype=float)

        def apply_cleanup(t_arr: np.ndarray, R_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            t_clean = np.asarray(t_arr, dtype=float).reshape(-1)
            R_clean = np.asarray(R_arr, dtype=float).reshape(-1)
            if t_clean.shape[0] != R_clean.shape[0]:
                return t_clean, R_clean
            mask = np.ones(R_clean.shape[0], dtype=bool)
            if chk_remove_below.isChecked():
                mask &= R_clean >= 0.0
            if chk_remove_spikes.isChecked() and R_clean.size >= 3:
                kernel = min(5, R_clean.size if R_clean.size % 2 == 1 else R_clean.size - 1)
                if kernel >= 3:
                    R_med = medfilt(R_clean.astype(float), kernel_size=kernel)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        ratio = np.where(R_med > 0, R_clean / R_med, 1.0)
                    mask &= ratio <= float(spin_spike_threshold.value())
            return t_clean[mask], R_clean[mask]

        def selected_scalar(name: str):
            combo, preview = scalar_rows[name]
            key = combo.currentData()
            manual_edited = bool(preview.property("manual_edited"))
            text = preview.text().strip()
            if name == "legend":
                return self._normalise_legend_text(text) if text else None
            if name == "Req" and not key and not manual_edited:
                return None
            if text:
                try:
                    val = float(text.replace(",", ""))
                except ValueError as exc:
                    raise ValueError(f"{name} preview value must be numeric.") from exc
            else:
                key = combo.currentData()
                if not key:
                    return None
                val = self._wizard_to_float(ctx["flat"][key], None)
            converted = to_si(name, val)
            if converted is None:
                return None
            try:
                return float(np.asarray(converted).reshape(-1)[0])
            except Exception:
                return converted

        def apply_import():
            try:
                path = str(ctx["path"])
                has_time_mapping = bool(array_rows.get("t_exp") and array_rows["t_exp"].currentData())
                if (
                    has_time_mapping
                    and unit_for_preview("R_exp") == "pixel"
                    and not ctx.get("pixel_time_warning_accepted", False)
                ):
                    reply = QMessageBox.warning(
                        dlg,
                        "Check R_exp unit",
                        "A time-axis variable is selected, but R_exp is set to pixel.\n\n"
                        "If this file already contains physical radius values, choose m or um "
                        "instead. Continue importing with pixel-to-um conversion?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
                    ctx["pixel_time_warning_accepted"] = True
                t = selected_array("t_exp")
                R = selected_array("R_exp")
                if t is None or R is None:
                    if R is None:
                        raise ValueError("Please map R_exp before importing.")
                    fps = float(spin_fps.value())
                    if not np.isfinite(fps) or fps <= 0:
                        raise ValueError("fps must be positive to reconstruct t_exp.")
                    t = np.arange(R.shape[0], dtype=float) / fps
                if t.shape[0] != R.shape[0]:
                    raise ValueError(f"t_exp and R_exp length mismatch: {t.shape[0]} vs {R.shape[0]}.")
                t, R = apply_cleanup(t, R)
                legend = selected_scalar("legend") or Path(path).stem
                t_sim = selected_array("t_sim")
                R_sim = selected_array("R_sim")
                t_key = array_rows["t_exp"].currentData() or ""
                R_key = array_rows["R_exp"].currentData() or ""
                import_meta = {
                    "method": "import_wizard",
                    "source_path": path,
                    "t_key": str(t_key),
                    "R_key": str(R_key),
                    "t_unit": unit_for_preview("t_exp"),
                    "R_unit": unit_for_preview("R_exp"),
                    "used_fps": not bool(t_key),
                    "fps": float(spin_fps.value()),
                    "um_per_pixel": float(spin_um_per_pixel.value()),
                    "remove_negative_R": bool(chk_remove_below.isChecked()),
                    "remove_isolated_spikes": bool(chk_remove_spikes.isChecked()),
                    "spike_threshold": float(spin_spike_threshold.value()),
                }
                self._apply_import_wizard_experiment(
                    path=path,
                    t=t,
                    R=R,
                    legend=str(legend),
                    Req=selected_scalar("Req"),
                    P_inf=selected_scalar("P_inf"),
                    rho=selected_scalar("rho"),
                    c_long=selected_scalar("c_long"),
                    gamma=selected_scalar("gamma"),
                    t_sim=t_sim,
                    R_sim=R_sim,
                    import_metadata=import_meta,
                )
                dlg.close()
            except Exception as exc:
                QMessageBox.critical(dlg, "Import failed", f"{exc}\n\n{traceback.format_exc()}")

        def current_keyword_defaults() -> dict[str, list[str]]:
            updated = self._normalise_import_wizard_keywords(self._import_wizard_keywords)
            for name, combo in array_rows.items():
                selected = combo.currentData()
                if not selected:
                    continue
                current = updated.get(name, [])
                updated[name] = [str(selected)] + [v for v in current if v != str(selected)]
            for name, (combo, _preview) in scalar_rows.items():
                selected = combo.currentData()
                if not selected:
                    continue
                current = updated.get(name, [])
                updated[name] = [str(selected)] + [v for v in current if v != str(selected)]
            return updated

        def set_wizard_defaults():
            self._import_wizard_keywords = current_keyword_defaults()
            self._import_wizard_units = self._normalise_import_wizard_units({
                name: str(combo.currentData() or "")
                for name, combo in unit_rows.items()
            })
            self._import_wizard_um_per_pixel = float(spin_um_per_pixel.value())
            self._import_wizard_fps = float(spin_fps.value())
            self._import_wizard_remove_below = bool(chk_remove_below.isChecked())
            self._import_wizard_remove_spikes = bool(chk_remove_spikes.isChecked())
            self._import_wizard_spike_threshold = float(spin_spike_threshold.value())
            self._save_settings()
            self.statusBar().showMessage("Import Wizard keywords saved as default.")

        def reset_wizard_defaults():
            self._import_wizard_keywords = self._normalise_import_wizard_keywords({})
            self._import_wizard_units = self._normalise_import_wizard_units({})
            self._import_wizard_um_per_pixel = 3.2
            self._import_wizard_fps = 1_000_000.0
            self._import_wizard_remove_below = False
            self._import_wizard_remove_spikes = False
            self._import_wizard_spike_threshold = 2.0
            spin_um_per_pixel.setValue(float(self._import_wizard_um_per_pixel))
            spin_fps.setValue(float(self._import_wizard_fps))
            chk_remove_below.setChecked(bool(self._import_wizard_remove_below))
            chk_remove_spikes.setChecked(bool(self._import_wizard_remove_spikes))
            spin_spike_threshold.setValue(float(self._import_wizard_spike_threshold))
            self._save_settings()
            if ctx["path"]:
                try:
                    rebuild_for_path(str(ctx["path"]))
                except Exception as exc:
                    QMessageBox.critical(dlg, "Import failed", f"{exc}\n\n{traceback.format_exc()}")
            self.statusBar().showMessage("Import Wizard keywords reset.")

        btn_import.clicked.connect(choose_file)
        drop.fileDropped.connect(dropped_file)
        btn_apply.clicked.connect(apply_import)
        btn_set_default.clicked.connect(set_wizard_defaults)
        btn_reset.clicked.connect(reset_wizard_defaults)

        def toggle_micro_units():
            using_micro_units["value"] = not using_micro_units["value"]
            target = {
                "t_exp": "us",
                "t_sim": "us",
                "R_exp": "um",
                "R_sim": "um",
                "Req": "um",
                "Rmax": "um",
            } if using_micro_units["value"] else {
                "t_exp": "s",
                "t_sim": "s",
                "R_exp": "m",
                "R_sim": "m",
                "Req": "m",
                "Rmax": "m",
            }
            for name, unit in target.items():
                combo = unit_rows.get(name)
                if combo is None:
                    continue
                idx = combo.findData(unit)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            btn_units.setText("Use SI units" if using_micro_units["value"] else "Use um/us")

        btn_units.clicked.connect(toggle_micro_units)

        self._import_wizard_dialog = dlg
        if path:
            try:
                rebuild_for_path(path)
            except Exception as exc:
                QMessageBox.critical(dlg, "Import failed", f"{exc}\n\n{traceback.format_exc()}")
        dlg.show()

    # =====================================================================
    # load experiment
    # =====================================================================

    def on_load_experiment(self):
        if self.state.mode == "jobs":
            QMessageBox.information(
                self,
                "Job List active",
                "Switch to Simulation or Fitting mode before loading experiment data.",
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load experiment .mat", "", "MAT files (*.mat)"
        )
        if not path:
            return
        try:
            try:
                exp = self._load_experiment_with_import_defaults(
                    path,
                    confirm_units=True,
                    parent=self,
                    accepted_state={},
                )
            except RuntimeError:
                return
            except Exception:
                exp = load_experiment_mat(path)
            self.state.exp_t = exp.t
            self.state.exp_R = exp.R
            self.state.exp_path = exp.source_path
            self.state.import_metadata = getattr(exp, "import_metadata", None)
            self.state.P_inf = exp.P_inf
            self.state.rho = exp.rho
            self.state.R_eq = exp.R_eq
            self._editing_job_index = None
            if self.state.mode == "fitting":
                self.btn_add_job.setText("Add to job list")
                self.btn_add_job.setToolTip("Add the current fitting setup as a queued job.")
            self._update_window_title()

            self.state.sim_t = None
            self.state.sim_R = None
            self.state.sim_meta = None
            self.state.best_fit_t = None
            self.state.best_fit_R = None
            self.state.best_fit_meta = None

            if self.state.R_eq is None and exp.R.size > 0:
                self.state.R_eq = float(np.mean(exp.R[-min(20, exp.R.size):]))

            if self.state.R_eq is not None:
                self.spin_Req_um.setValue(self.state.R_eq * 1e6)

            file_info = []
            if exp.P_inf is not None:
                file_info.append(f"P_inf={exp.P_inf:.1f}")
            if exp.rho is not None:
                file_info.append(f"rho={exp.rho:.1f}")

            mismatch_parts = []
            if exp.P_inf is not None and abs(exp.P_inf - self.spin_P_inf.value()) > 1.0:
                mismatch_parts.append(
                    f"P_inf: file={exp.P_inf:.1f}, GUI={self.spin_P_inf.value():.1f}"
                )
            if exp.rho is not None and abs(exp.rho - self.spin_rho.value()) > 0.1:
                mismatch_parts.append(
                    f"rho: file={exp.rho:.1f}, GUI={self.spin_rho.value():.1f}"
                )
            if mismatch_parts:
                QMessageBox.information(
                    self, "P_inf / rho in file",
                    "The loaded .mat file contains values that differ from "
                    "the current GUI settings:\n\n"
                    + "\n".join(mismatch_parts) + "\n\n"
                    "The GUI values were NOT changed. "
                    "Update the P_inf / rho spinners manually if needed.",
                )

            if self.state.exp_t is not None and self.state.exp_t.size > 0:
                self.spin_t_fit_start.setValue(float(self.state.exp_t[0]) * 1e6)
                self.spin_t_fit_end.setValue(float(self.state.exp_t[-1]) * 1e6)
                if self.chk_fit_window_cycles.isChecked():
                    self._apply_fit_window_cycles()

            if self._curve_view_active():
                curve_legend = Path(path).stem
                try:
                    curve_legend = self._curve_legend_from_mat(
                        loadmat(path, squeeze_me=True, struct_as_record=False),
                        curve_legend,
                    )
                except Exception:
                    curve_legend = self._normalise_legend_text(curve_legend)
                self._add_curve_to_view(
                    curve_type="experiment",
                    t=exp.t,
                    R=exp.R,
                    legend=curve_legend,
                )

            extra = ""
            if file_info:
                extra = f"  [file contains: {', '.join(file_info)}]"
            self.statusBar().showMessage(
                f"Loaded exp data: t='{exp.t_key}', R='{exp.R_key}' from {path}{extra}"
            )
            self._redraw_all()
        except ValueError as e:
            err_text = str(e)
            if "struct_best_fit" in err_text:
                reply = QMessageBox.question(
                    self, "Not an experiment file",
                    "This .mat file contains fitting parameters (struct_best_fit), "
                    "not experimental R(t) data.\n\n"
                    "Load it as parameters instead?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._load_params_from_path(path)
            else:
                reply = QMessageBox.question(
                    self,
                    "Load failed",
                    f"{e}\n\nOpen Import Wizard to map variables manually?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.on_import_wizard(path)
        except Exception as e:
            reply = QMessageBox.question(
                self,
                "Load failed",
                f"{e}\n\nOpen Import Wizard to map variables manually?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.on_import_wizard(path)

    # =====================================================================
    # job queue
    # =====================================================================

    @staticmethod
    def _job_status_color(status: str) -> QColor:
        colors = {
            "queued": QColor(230, 230, 230),
            "running": QColor(255, 244, 179),
            "completed": QColor(205, 239, 211),
            "failed": QColor(245, 204, 204),
        }
        return colors.get(status, QColor(255, 255, 255))

    @staticmethod
    def _job_model_label(model_key: str) -> str:
        labels = {"NHKV (Rmax)": "Rmax"}
        return labels.get(model_key, model_key)

    def _csv_param_label(self, model_key: str, param_name: str) -> str:
        model = self._get_model_definition(model_key)
        if model is None:
            return param_name
        for param in model.parameters:
            if param.name != param_name:
                continue
            if param.units:
                unit = str(param.units[0].label).strip()
                if unit:
                    return f"{param_name} ({unit})"
            return param_name
        return param_name

    @staticmethod
    def _split_cleanup_patterns(patterns: str) -> list[str]:
        return [p.strip() for p in str(patterns or "").split(";") if p.strip()]

    def _clean_imported_experiment_name(self, name: str) -> str:
        text = str(name or "")
        if not self._job_import_name_cleanup_enabled:
            return text
        patterns = self._split_cleanup_patterns(self._job_import_name_cleanup_regex)
        if not patterns:
            return text
        cleaned = text
        try:
            for pattern in patterns:
                cleaned = re.sub(pattern, "", cleaned)
        except re.error:
            return text
        cleaned = cleaned.strip(" _-.")
        suffix = Path(text).suffix
        if suffix and not cleaned.lower().endswith(suffix.lower()):
            cleaned = f"{cleaned}{suffix}"
        return cleaned or text

    def _selected_job_index(self) -> int | None:
        ranges = self.tbl_jobs.selectedRanges()
        if not ranges:
            return None
        row = ranges[0].topRow()
        if 0 <= row < len(self._jobs):
            return row
        return None

    def _job_tooltip(self, job: dict) -> str:
        exp = job.get("experiment", {})
        fit_window = job.get("fit_window", {})
        lines = [
            f"Status: {job.get('status', 'queued')}",
            f"Experiment: {exp.get('file_name', '')}",
            f"Model: {job.get('model', '')}",
            f"Fit window: {fit_window.get('t_start_s', 0.0) * 1e6:.3f} to "
            f"{fit_window.get('t_end_s', 0.0) * 1e6:.3f} us",
            f"Points: {fit_window.get('n_points', 0)}",
        ]
        if job.get("lsq_err") is not None:
            lines.append(f"LSQErr: {job['lsq_err']:.6g}")
        best_params = job.get("best_params")
        if isinstance(best_params, dict) and best_params:
            lines.append("Best-fit parameters:")
            for name, value in best_params.items():
                try:
                    lines.append(f"  {name} = {float(value):.6g}")
                except Exception:
                    lines.append(f"  {name} = {value}")
        elif job.get("error"):
            lines.append(f"Error: {job['error']}")
        return "\n".join(lines)

    def _refresh_job_table(self):
        self.tbl_jobs.setRowCount(len(self._jobs))
        for row, job in enumerate(self._jobs):
            exp = job.get("experiment", {})
            values = [
                str(row + 1),
                job.get("type", "fit").title(),
                exp.get("file_name", ""),
                self._job_model_label(job.get("model", "")),
            ]
            bg = self._job_status_color(job.get("status", "queued"))
            tooltip = self._job_tooltip(job)
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setToolTip(tooltip)
                item.setBackground(bg)
                item.setForeground(QColor(0, 0, 0))
                self.tbl_jobs.setItem(row, col, item)
        self._update_job_buttons()

    def _update_job_buttons(self):
        idx = self._selected_job_index()
        selected = idx is not None
        selected_status = self._jobs[idx].get("status", "queued") if selected else ""
        editable_statuses = ("queued", "failed", "completed")
        can_reorder = selected and selected_status == "queued" and not self._queue_running
        queued_indices = [
            i for i, job in enumerate(self._jobs) if job.get("status", "queued") == "queued"
        ]
        first_queued = queued_indices[0] if queued_indices else None
        last_queued = queued_indices[-1] if queued_indices else None
        self.btn_job_top.setEnabled(can_reorder and first_queued is not None and idx != first_queued)
        self.btn_job_up.setEnabled(can_reorder and first_queued is not None and idx > first_queued)
        self.btn_job_down.setEnabled(can_reorder and last_queued is not None and idx < last_queued)
        self.btn_remove_job.setEnabled(selected and selected_status in editable_statuses)
        self.btn_load_job.setEnabled(selected and selected_status != "running")
        self.btn_clear_completed_jobs.setEnabled(
            (not self._queue_running)
            and any(j.get("status") == "completed" for j in self._jobs)
        )
        self.btn_run_jobs.setEnabled(
            (not self._queue_running)
            and any(j.get("status") == "queued" for j in self._jobs)
        )
        self.btn_stop_jobs.setEnabled(self._queue_running and not self._queue_stop_after_current)

    def _confirm_initial_values_within_bounds(
        self,
        initial_values: dict[str, float],
        bounds_si: dict[str, tuple[float, float]],
        fit_flags: dict[str, bool],
    ) -> bool:
        rows: list[str] = []
        for name, enabled in fit_flags.items():
            if not enabled or name not in initial_values or name not in bounds_si:
                continue
            value = float(initial_values[name])
            lb, ub = bounds_si[name]
            lb = float(lb)
            ub = float(ub)
            if lb > ub:
                lb, ub = ub, lb
            if value < lb or value > ub:
                rows.append(
                    f"{name}: current={value:.6g}, bounds=[{lb:.6g}, {ub:.6g}]"
                )

        if not rows:
            return True

        reply = QMessageBox.question(
            self,
            "Initial value outside bounds",
            "Some fitted parameters start outside their bounds:\n\n"
            + "\n".join(rows)
            + "\n\nContinue anyway? The optimizer will start from the nearest bound.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _build_fit_job_snapshot_from_data(
        self,
        *,
        exp_t: np.ndarray,
        exp_R: np.ndarray,
        exp_path: str,
        exp_name: str,
        exp_P_inf: float | None,
        exp_rho: float | None,
        exp_R_eq: float | None,
        exp_import_metadata: dict | None = None,
        fit_flags: dict[str, bool],
        scales: dict[str, str],
        bounds_si: dict[str, tuple[float, float]],
    ) -> tuple[dict | None, str | None]:
        if exp_t is None or exp_R is None or exp_t.size == 0 or exp_R.size == 0:
            return None, "Experiment data is empty."
        if exp_t.shape[0] != exp_R.shape[0]:
            return None, f"t/R length mismatch: {exp_t.shape[0]} vs {exp_R.shape[0]}."
        if not any(fit_flags.values()):
            return None, "No parameters are selected for fitting."

        model_key = self._get_active_model_key()
        t_start_s, t_end_s, cycles = self._current_fit_window_seconds(
            exp_t, exp_R, model_key
        )
        window_mode = "auto_cycles" if self.chk_fit_window_cycles.isChecked() else "manual"

        mask = (exp_t >= t_start_s) & (exp_t <= t_end_s)
        n_points = int(np.count_nonzero(mask))
        if n_points < 3:
            return None, f"Fit window contains fewer than 3 points ({n_points})."

        req_m = (
            float(exp_R_eq)
            if exp_R_eq is not None and np.isfinite(float(exp_R_eq)) and float(exp_R_eq) > 0.0
            else float(self.spin_Req_um.value()) * 1e-6
        )

        return {
            "version": 1,
            "type": "fit",
            "status": "queued",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": {
                "path": exp_path,
                "file_name": exp_name,
                "t": np.array(exp_t, dtype=float).copy(),
                "R": np.array(exp_R, dtype=float).copy(),
                "P_inf": exp_P_inf,
                "rho": exp_rho,
                "R_eq": exp_R_eq,
                "import_metadata": dict(exp_import_metadata or {}),
            },
            "model": model_key,
            "solver_entrypoint": self._plugin_entrypoint_for_model(model_key),
            "constants": dict(self._model_constants),
            "parameters": self._collect_parameter_defaults(),
            "initial_values": self._get_param_si(),
            "fit_flags": dict(fit_flags),
            "scales": dict(scales),
            "bounds_si": dict(bounds_si),
            "experiment_settings": {
                "Req_um": float(req_m) * 1e6,
                "tspan_us": float(self.spin_tspan_us.value()),
            },
            "fit_window": {
                "mode": window_mode,
                "cycles": cycles,
                "t_start_s": float(t_start_s),
                "t_end_s": float(t_end_s),
                "n_points": n_points,
            },
            "physics": {
                "bubble_model": self._bubble_model,
                "P_inf": float(self.spin_P_inf.value()),
                "rho": float(self.spin_rho.value()),
                "c_long": float(self.spin_c_long.value()),
                "gamma": float(self.spin_gamma.value()),
                "NT": int(self.spin_NT.value()),
                **self._get_solver_settings(),
            },
            "optimizer": asdict(self._opt_config),
            "result": None,
            "lsq_err": None,
            "error": None,
        }, None

    def _on_job_selection_changed(self):
        idx = self._selected_job_index()
        if idx is not None:
            self.tbl_jobs.blockSignals(True)
            self.tbl_jobs.selectRow(idx)
            self.tbl_jobs.blockSignals(False)
        self._update_job_buttons()
        if self.state.mode == "jobs":
            self._redraw_all()

    def _build_fit_job_snapshot(self) -> dict | None:
        if self.state.exp_t is None or self.state.exp_R is None:
            QMessageBox.warning(self, "No data", "Please load experiment data before adding a fit job.")
            return None

        fit_flags, scales, bounds_si = self._collect_fit_setup()
        if not any(fit_flags.values()):
            QMessageBox.warning(
                self, "No parameters selected",
                "Enable the Fit checkbox for at least one parameter.",
            )
            return None

        initial_values = self._get_param_si()
        if not self._confirm_initial_values_within_bounds(
            initial_values, bounds_si, fit_flags
        ):
            return None

        if self.chk_fit_window_cycles.isChecked():
            self._apply_fit_window_cycles()

        exp_path = self.state.exp_path or ""
        exp_name = Path(exp_path).name if exp_path else "loaded experiment"
        if self._editing_job_index is not None and 0 <= self._editing_job_index < len(self._jobs):
            existing_exp = self._jobs[self._editing_job_index].get("experiment", {})
            exp_name = existing_exp.get("file_name") or exp_name
        job, error = self._build_fit_job_snapshot_from_data(
            exp_t=self.state.exp_t,
            exp_R=self.state.exp_R,
            exp_path=exp_path,
            exp_name=exp_name,
            exp_P_inf=self.state.P_inf,
            exp_rho=self.state.rho,
            exp_R_eq=self.state.R_eq,
            exp_import_metadata=self.state.import_metadata,
            fit_flags=fit_flags,
            scales=scales,
            bounds_si=bounds_si,
        )
        if job is None:
            QMessageBox.warning(self, "Cannot add job", error or "Could not build fit job.")
        return job

    def _build_sim_job_snapshot(self) -> dict | None:
        if self.chk_fit_window_cycles.isChecked() and self.state.exp_t is not None:
            self._apply_fit_window_cycles()

        exp_path = self.state.exp_path or ""
        exp_name = Path(exp_path).name if exp_path else "simulation"
        job, error = self._build_sim_job_snapshot_from_data(
            exp_t=self.state.exp_t,
            exp_R=self.state.exp_R,
            exp_path=exp_path,
            exp_name=exp_name,
            exp_P_inf=self.state.P_inf,
            exp_rho=self.state.rho,
            exp_R_eq=self.state.R_eq,
            exp_import_metadata=self.state.import_metadata,
        )
        if job is None:
            QMessageBox.warning(self, "Cannot add job", error or "Could not build simulation job.")
        return job

    def _build_sim_job_snapshot_from_data(
        self,
        *,
        exp_t: np.ndarray,
        exp_R: np.ndarray,
        exp_path: str,
        exp_name: str,
        exp_P_inf: float | None,
        exp_rho: float | None,
        exp_R_eq: float | None,
        exp_import_metadata: dict | None = None,
    ) -> tuple[dict | None, str | None]:
        model_key = self._get_active_model_key()
        has_exp = exp_t is not None and exp_R is not None
        exp_t_arr = np.asarray(exp_t, dtype=float).reshape(-1) if has_exp else np.array([], dtype=float)
        exp_R_arr = np.asarray(exp_R, dtype=float).reshape(-1) if has_exp else np.array([], dtype=float)
        if exp_t_arr.size >= 3 and exp_R_arr.size >= 3 and exp_t_arr.shape[0] == exp_R_arr.shape[0]:
            t_start_s, t_end_s, cycles = self._current_fit_window_seconds(exp_t_arr, exp_R_arr, model_key)
            window_mode = "auto_cycles" if self.chk_fit_window_cycles.isChecked() else "manual"
            mask = (exp_t_arr >= t_start_s) & (exp_t_arr <= t_end_s)
            n_points = int(np.count_nonzero(mask))
            if n_points < 3:
                return None, f"Simulation LSQErr window contains fewer than 3 points ({n_points})."
        elif exp_t_arr.size == 0 and exp_R_arr.size == 0:
            t_start_s = np.nan
            t_end_s = np.nan
            cycles = None
            window_mode = "none"
            n_points = 0
        else:
            return None, "Experiment data is empty or invalid."

        fit_flags, scales, bounds_si = self._collect_fit_setup()
        req_m = (
            float(exp_R_eq)
            if exp_R_eq is not None and np.isfinite(float(exp_R_eq)) and float(exp_R_eq) > 0.0
            else float(self.spin_Req_um.value()) * 1e-6
        )
        return {
            "version": 1,
            "type": "simulation",
            "status": "queued",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": {
                "path": exp_path,
                "file_name": exp_name,
                "t": exp_t_arr.copy(),
                "R": exp_R_arr.copy(),
                "P_inf": exp_P_inf,
                "rho": exp_rho,
                "R_eq": exp_R_eq,
                "import_metadata": dict(exp_import_metadata or {}),
            },
            "model": model_key,
            "solver_entrypoint": self._plugin_entrypoint_for_model(model_key),
            "constants": dict(self._model_constants),
            "parameters": self._collect_parameter_defaults(),
            "initial_values": self._get_param_si(),
            "fit_flags": dict(fit_flags),
            "scales": dict(scales),
            "bounds_si": dict(bounds_si),
            "experiment_settings": {
                "Req_um": float(req_m) * 1e6,
                "tspan_us": float(self.spin_tspan_us.value()),
            },
            "fit_window": {
                "mode": window_mode,
                "cycles": cycles,
                "t_start_s": float(t_start_s),
                "t_end_s": float(t_end_s),
                "n_points": n_points,
            },
            "physics": {
                "bubble_model": self._bubble_model,
                "P_inf": float(self.spin_P_inf.value()),
                "rho": float(self.spin_rho.value()),
                "c_long": float(self.spin_c_long.value()),
                "gamma": float(self.spin_gamma.value()),
                "NT": int(self.spin_NT.value()),
                **self._get_solver_settings(),
            },
            "optimizer": {},
            "result": None,
            "lsq_err": None,
            "error": None,
        }, None

    def on_add_job(self):
        if self.state.mode == "fitting":
            job = self._build_fit_job_snapshot()
        elif self.state.mode == "simulation":
            job = self._build_sim_job_snapshot()
        else:
            QMessageBox.information(
                self, "Editor mode required",
                "Switch to Simulation or Fitting mode before adding a job.",
            )
            return
        if job is None:
            return
        if self._editing_job_index is not None:
            idx = self._editing_job_index
            if 0 <= idx < len(self._jobs) and self._jobs[idx].get("status") in ("queued", "failed"):
                self._jobs[idx] = job
                self._refresh_job_table()
                self._select_job_row(idx)
                self._editing_job_index = None
                self.btn_add_job.setText("Add to job list")
                self.btn_add_job.setToolTip("Add the current fitting setup as a queued job.")
                self.state.sim_t = None
                self.state.sim_R = None
                self.state.sim_meta = None
                self.state.best_fit_t = None
                self.state.best_fit_R = None
                self.state.best_fit_meta = None
                self._set_mode("jobs")
                self._select_job_row(idx)
                self.statusBar().showMessage(f"Updated job {idx + 1}: {job['experiment']['file_name']}")
                return
            self._editing_job_index = None
            self.btn_add_job.setText("Add to job list")

        self._jobs.append(job)
        self._refresh_job_table()
        self.statusBar().showMessage(
            f"Added job {len(self._jobs)}: {job['experiment']['file_name']}"
        )

    def on_batch_add_experiments_as_jobs(self):
        if self.state.mode not in ("fitting", "simulation"):
            QMessageBox.information(
                self,
                "Editor mode required",
                "Switch to Simulation or Fitting mode before batch-adding experiments as jobs.",
            )
            return

        is_fitting_batch = self.state.mode == "fitting"
        fit_flags, scales, bounds_si = self._collect_fit_setup()
        if is_fitting_batch and not any(fit_flags.values()):
            QMessageBox.warning(
                self,
                "No parameters selected",
                "Enable the Fit checkbox for at least one parameter before batch-adding jobs.",
            )
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Batch add experiments as jobs",
            "",
            "MAT files (*.mat)",
        )
        if not paths:
            return

        added: list[dict] = []
        skipped: list[str] = []
        progress = QProgressDialog("Preparing batch jobs...", "", 0, len(paths), self)
        progress.setWindowTitle("Batch add jobs")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        try:
            import_unit_warning_state: dict = {}
            for i, path in enumerate(paths, start=1):
                progress.setLabelText(f"Loading {i}/{len(paths)}: {Path(path).name}")
                progress.setValue(i - 1)
                QApplication.processEvents()
                try:
                    try:
                        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
                        result_job = self._job_from_result_mat(path, mat)
                    except Exception:
                        result_job = None
                    if result_job is not None:
                        added.append(result_job)
                        continue

                    try:
                        exp = self._load_experiment_with_import_defaults(
                            path,
                            confirm_units=True,
                            parent=self,
                            accepted_state=import_unit_warning_state,
                        )
                    except RuntimeError:
                        skipped.append(f"{Path(path).name}: import cancelled by user")
                        continue
                    except Exception:
                        exp = load_experiment_mat(path)
                    exp_R_eq = exp.R_eq
                    if exp_R_eq is None and exp.R.size > 0:
                        exp_R_eq = float(np.mean(exp.R[-min(20, exp.R.size):]))
                    exp_name = self._clean_imported_experiment_name(Path(exp.source_path).name)
                    if is_fitting_batch:
                        job, error = self._build_fit_job_snapshot_from_data(
                            exp_t=exp.t,
                            exp_R=exp.R,
                            exp_path=exp.source_path,
                            exp_name=exp_name,
                            exp_P_inf=exp.P_inf,
                            exp_rho=exp.rho,
                            exp_R_eq=exp_R_eq,
                            exp_import_metadata=getattr(exp, "import_metadata", None),
                            fit_flags=fit_flags,
                            scales=scales,
                            bounds_si=bounds_si,
                        )
                    else:
                        job, error = self._build_sim_job_snapshot_from_data(
                            exp_t=exp.t,
                            exp_R=exp.R,
                            exp_path=exp.source_path,
                            exp_name=exp_name,
                            exp_P_inf=exp.P_inf,
                            exp_rho=exp.rho,
                            exp_R_eq=exp_R_eq,
                            exp_import_metadata=getattr(exp, "import_metadata", None),
                        )
                    if job is None:
                        skipped.append(f"{Path(path).name}: {error or 'could not build job'}")
                        continue
                    added.append(job)
                except Exception as e:
                    skipped.append(f"{Path(path).name}: {e}")
                finally:
                    progress.setValue(i)
                    QApplication.processEvents()
        finally:
            progress.close()

        if added:
            first_idx = len(self._jobs)
            self._jobs.extend(added)
            self._refresh_job_table()
            self._set_mode("jobs")
            self._select_job_row(first_idx)

        if skipped:
            msg = f"Added {len(added)} job(s), skipped {len(skipped)} file(s)."
            detail = "\n".join(skipped[:20])
            if len(skipped) > 20:
                detail += f"\n... and {len(skipped) - 20} more"
            QMessageBox.warning(self, "Batch add completed with skipped files", f"{msg}\n\n{detail}")
        else:
            self.statusBar().showMessage(f"Batch added {len(added)} job(s).")

    def _parse_sweep_values(self, text: str) -> list[float]:
        raw = str(text or "").strip()
        if not raw:
            return []

        safe_globals = {
            "__builtins__": {},
            "np": np,
            "numpy": np,
            "array": np.array,
            "linspace": np.linspace,
            "arange": np.arange,
            "logspace": np.logspace,
            "geomspace": np.geomspace,
        }
        try:
            value = eval(raw, safe_globals, {})
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            parts = [p for p in re.split(r"[,;\s]+", raw) if p]
            arr = np.asarray([float(p) for p in parts], dtype=float)
        values = [float(v) for v in arr if np.isfinite(float(v))]
        if not values:
            raise ValueError("No finite numeric values were found.")
        return values

    @staticmethod
    def _sweep_legend_name(param_name: str) -> str:
        greek = {
            "alpha": r"\alpha",
            "alphaA": r"\alphaA",
            "alphaB": r"\alphaB",
            "alphaC": r"\alphaC",
            "beta": r"\beta",
            "gamma": r"\gamma",
            "lambda": r"\lambda",
            "lambda_Y": r"\lambda_Y",
            "mu": r"\mu",
            "muB": r"\muB",
            "muC": r"\muC",
        }
        return greek.get(param_name, param_name)

    def _sweep_legend_text(self, param_name: str, display_value: float, unit_label: str = "") -> str:
        name = self._sweep_legend_name(param_name)
        unit = str(unit_label or "").strip()
        if unit:
            return f"{name} = {display_value:.6g} {unit}"
        return f"{name} = {display_value:.6g}"

    def on_batch_create_simulation_jobs(self):
        if self.state.mode != "simulation":
            QMessageBox.information(
                self,
                "Simulation mode required",
                "Switch to Simulation mode before creating parameter-sweep simulation jobs.",
            )
            return
        if self._current_model is None or not self._param_rows:
            QMessageBox.warning(self, "No model", "No active constitutive model is loaded.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Batch create simulation jobs")
        dialog.resize(760, 420)
        layout = QVBoxLayout(dialog)

        top = QHBoxLayout()
        title = QLabel(
            "Create simulation jobs by sweeping one parameter at a time from the same base values."
        )
        top.addWidget(title, stretch=1)
        help_btn = QToolButton()
        help_btn.setText("?")
        help_btn.setToolTip(
            "Sweep values are entered in the displayed unit for that row.\n"
            "Examples:\n"
            "  1000, 2000, 5000\n"
            "  np.linspace(1000, 5000, 9)\n"
            "  logspace(3, 5, 5)\n"
            "Each generated job changes only that row's parameter; all other parameters use the base column."
        )
        top.addWidget(help_btn)
        layout.addLayout(top)

        sweep_rows = [("Req", "Req", "um", float(self.spin_Req_um.value()), True)]
        for param in self._current_model.parameters:
            row_state = self._param_rows.get(param.name, {})
            unit_options = row_state.get("unit_options") or param.units or []
            unit_idx = int(row_state.get("unit_index", 0))
            unit_label = ""
            if unit_options and 0 <= unit_idx < len(unit_options):
                unit_label = str(unit_options[unit_idx].label)
            spin = row_state.get("spin")
            base_value = float(spin.value()) if spin is not None else float(param.default)
            sweep_rows.append((param.name, param.label or param.name, unit_label, base_value, False))

        table = QTableWidget(len(sweep_rows), 4, dialog)
        table.setHorizontalHeaderLabels(["Parameter", "Base", "Unit", "Sweep values"])
        table.verticalHeader().setVisible(False)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        base_edits: dict[str, QLineEdit] = {}
        sweep_edits: dict[str, QLineEdit] = {}
        for row_idx, (param_name, param_label, unit_label, base_value, _is_req) in enumerate(sweep_rows):
            name_item = QTableWidgetItem(param_label)
            name_item.setData(Qt.ItemDataRole.UserRole, param_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row_idx, 0, name_item)

            base_edit = QLineEdit(f"{base_value:.10g}")
            base_edit.setToolTip("Base value used for all jobs unless this row is being swept.")
            table.setCellWidget(row_idx, 1, base_edit)
            base_edits[param_name] = base_edit

            unit_item = QTableWidgetItem(unit_label)
            unit_item.setFlags(unit_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row_idx, 2, unit_item)

            sweep_edit = QLineEdit()
            sweep_edit.setPlaceholderText("e.g. 1, 2, 5 or np.linspace(1, 5, 9)")
            table.setCellWidget(row_idx, 3, sweep_edit)
            sweep_edits[param_name] = sweep_edit

        layout.addWidget(table)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            base_display: dict[str, float] = {
                name: float(edit.text().strip())
                for name, edit in base_edits.items()
            }
            sweeps: dict[str, list[float]] = {}
            for name, edit in sweep_edits.items():
                raw = edit.text().strip()
                if raw:
                    sweeps[name] = self._parse_sweep_values(raw)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid sweep values", str(exc))
            return

        if not sweeps:
            QMessageBox.information(
                self,
                "No sweep values",
                "Enter sweep values for at least one parameter.",
            )
            return

        if self.chk_fit_window_cycles.isChecked() and self.state.exp_t is not None:
            self._apply_fit_window_cycles()

        exp_path = self.state.exp_path or ""
        exp_name = Path(exp_path).name if exp_path else "simulation"
        param_defaults_base = self._collect_parameter_defaults()
        added: list[dict] = []
        skipped: list[str] = []

        unit_labels = {name: unit_label for name, _label, unit_label, _base, _is_req in sweep_rows}

        for sweep_name, values in sweeps.items():
            factor = self._get_unit_factor(sweep_name)
            for display_value in values:
                req_um = float(base_display.get("Req", float(self.spin_Req_um.value())))
                params_si = {
                    name: value * self._get_unit_factor(name)
                    for name, value in base_display.items()
                    if name != "Req"
                }
                if sweep_name == "Req":
                    req_um = float(display_value)
                else:
                    params_si[sweep_name] = float(display_value) * factor

                job, error = self._build_sim_job_snapshot_from_data(
                    exp_t=self.state.exp_t,
                    exp_R=self.state.exp_R,
                    exp_path=exp_path,
                    exp_name=exp_name,
                    exp_P_inf=self.state.P_inf,
                    exp_rho=self.state.rho,
                    exp_R_eq=self.state.R_eq,
                )
                if job is None:
                    skipped.append(f"{sweep_name}={display_value:g}: {error or 'could not build job'}")
                    continue

                param_defaults = {
                    name: dict(meta)
                    for name, meta in param_defaults_base.items()
                }
                for name, value_si in params_si.items():
                    if name in param_defaults:
                        param_defaults[name]["value_si"] = float(value_si)

                job["parameters"] = param_defaults
                job["initial_values"] = dict(params_si)
                job["best_params"] = dict(params_si)
                job["experiment_settings"]["Req_um"] = float(req_um)
                stem = Path(exp_name).stem or "simulation"
                suffix = self._safe_filename_part(f"{sweep_name}_{display_value:.6g}")
                job["experiment"]["file_name"] = f"{stem}__{suffix}.mat"
                job["legend"] = self._sweep_legend_text(
                    sweep_name,
                    float(display_value),
                    unit_labels.get(sweep_name, ""),
                )
                job["sweep"] = {
                    "parameter": sweep_name,
                    "value_display": float(display_value),
                    "value_si": float(req_um * 1e-6) if sweep_name == "Req" else float(params_si[sweep_name]),
                    "base_values_si": dict(params_si),
                    "Req_um": float(req_um),
                }
                added.append(job)

        if added:
            first_idx = len(self._jobs)
            self._jobs.extend(added)
            self._refresh_job_table()
            self._set_mode("jobs")
            self._select_job_row(first_idx)

        if skipped:
            msg = f"Created {len(added)} simulation job(s), skipped {len(skipped)} case(s)."
            detail = "\n".join(skipped[:20])
            if len(skipped) > 20:
                detail += f"\n... and {len(skipped) - 20} more"
            QMessageBox.warning(self, "Batch create completed with skipped cases", f"{msg}\n\n{detail}")
        else:
            self.statusBar().showMessage(f"Created {len(added)} simulation job(s).")

    def on_remove_selected_job(self):
        idx = self._selected_job_index()
        if idx is None:
            return
        if self._jobs[idx].get("status") not in ("queued", "failed", "completed"):
            QMessageBox.information(self, "Cannot remove", "Only queued, failed, or completed jobs can be removed.")
            return
        del self._jobs[idx]
        if self._editing_job_index == idx:
            self._editing_job_index = None
        elif self._editing_job_index is not None and self._editing_job_index > idx:
            self._editing_job_index -= 1
        self._refresh_job_table()

    def on_clear_completed_jobs(self):
        if self._queue_running:
            QMessageBox.information(
                self,
                "Queue running",
                "Wait for the queue to finish before clearing completed jobs.",
            )
            return
        before = len(self._jobs)
        self._jobs = [job for job in self._jobs if job.get("status") != "completed"]
        removed = before - len(self._jobs)
        self._editing_job_index = None
        self._refresh_job_table()
        self.statusBar().showMessage(f"Cleared {removed} completed job(s).")

    def _select_job_row(self, idx: int):
        if 0 <= idx < len(self._jobs):
            self.tbl_jobs.selectRow(idx)

    def _queued_indices(self) -> list[int]:
        return [
            i for i, job in enumerate(self._jobs)
            if job.get("status", "queued") == "queued"
        ]

    def _can_move_selected_job(self) -> int | None:
        idx = self._selected_job_index()
        if idx is None or self._queue_running:
            return None
        if self._jobs[idx].get("status") != "queued":
            return None
        return idx

    def on_move_job_up(self):
        idx = self._can_move_selected_job()
        queued = self._queued_indices()
        if idx is None or idx not in queued:
            return
        pos = queued.index(idx)
        if pos <= 0:
            return
        swap_idx = queued[pos - 1]
        self._jobs[swap_idx], self._jobs[idx] = self._jobs[idx], self._jobs[swap_idx]
        self._refresh_job_table()
        self._select_job_row(swap_idx)

    def on_move_job_down(self):
        idx = self._can_move_selected_job()
        queued = self._queued_indices()
        if idx is None or idx not in queued:
            return
        pos = queued.index(idx)
        if pos >= len(queued) - 1:
            return
        swap_idx = queued[pos + 1]
        self._jobs[swap_idx], self._jobs[idx] = self._jobs[idx], self._jobs[swap_idx]
        self._refresh_job_table()
        self._select_job_row(swap_idx)

    def on_move_job_to_top(self):
        idx = self._can_move_selected_job()
        queued = self._queued_indices()
        if idx is None or idx not in queued:
            return
        first = queued[0]
        if idx == first:
            return
        job = self._jobs.pop(idx)
        self._jobs.insert(first, job)
        self._refresh_job_table()
        self._select_job_row(first)

    def on_load_selected_job_to_editor(self):
        idx = self._selected_job_index()
        if idx is None:
            return
        job = self._jobs[idx]
        if job.get("status") == "running":
            QMessageBox.information(self, "Cannot load", "The running job cannot be loaded to editor.")
            return

        model_key = job.get("model", "")
        model_idx = self._cmb_model.findText(model_key)
        if model_idx < 0:
            QMessageBox.warning(self, "Unknown model", f"Cannot load unknown model: {model_key}")
            return

        self._cmb_model.setCurrentIndex(model_idx)
        self._apply_parameter_defaults(job.get("parameters", {}))
        best_params = job.get("best_params")
        if isinstance(best_params, dict) and best_params:
            for name, value in best_params.items():
                row = self._param_rows.get(name)
                if row is not None:
                    factor = self._get_unit_factor(name)
                    row["spin"].setValue(float(value) / factor)

        exp = job.get("experiment", {})
        self.state.exp_t = np.array(exp.get("t"), dtype=float).copy()
        self.state.exp_R = np.array(exp.get("R"), dtype=float).copy()
        self.state.exp_path = exp.get("path") or None
        self.state.P_inf = exp.get("P_inf")
        self.state.rho = exp.get("rho")
        self.state.R_eq = exp.get("R_eq")
        self._update_window_title()

        settings = job.get("experiment_settings", {})
        if settings.get("Req_um") is not None:
            self.spin_Req_um.setValue(float(settings["Req_um"]))
        if settings.get("tspan_us") is not None:
            self.spin_tspan_us.setValue(float(settings["tspan_us"]))

        fit_window = job.get("fit_window", {})
        self.chk_fit_window_cycles.setChecked(fit_window.get("mode") == "auto_cycles")
        if fit_window.get("cycles") is not None:
            self.spin_fit_window_cycles.setValue(int(fit_window["cycles"]))
        if fit_window.get("t_start_s") is not None:
            self.spin_t_fit_start.setValue(float(fit_window["t_start_s"]) * 1e6)
        if fit_window.get("t_end_s") is not None:
            self.spin_t_fit_end.setValue(float(fit_window["t_end_s"]) * 1e6)

        phys = job.get("physics", {})
        if phys.get("bubble_model"):
            self._bubble_model = str(phys["bubble_model"])
            self._act_rp.setChecked(self._bubble_model == "Rayleigh-Plesset")
            self._act_km.setChecked(self._bubble_model != "Rayleigh-Plesset")
        if phys.get("P_inf") is not None:
            self.spin_P_inf.setValue(float(phys["P_inf"]))
        if phys.get("rho") is not None:
            self.spin_rho.setValue(float(phys["rho"]))
        if phys.get("c_long") is not None:
            self.spin_c_long.setValue(float(phys["c_long"]))
        if phys.get("gamma") is not None:
            self.spin_gamma.setValue(float(phys["gamma"]))
        if phys.get("NT") is not None:
            self.spin_NT.setValue(int(phys["NT"]))
        if phys.get("solver_method"):
            solver_idx = self._cmb_solver.findText(str(phys["solver_method"]))
            if solver_idx >= 0:
                self._cmb_solver.setCurrentIndex(solver_idx)
        if phys.get("rel_tol") is not None:
            self.le_rtol.setText(str(phys["rel_tol"]))
        if phys.get("abs_tol") is not None:
            self.le_atol.setText(str(phys["abs_tol"]))

        if isinstance(job.get("optimizer"), dict):
            try:
                self._opt_config = OptConfig(**job["optimizer"])
            except TypeError:
                pass

        self.state.sim_t = None
        self.state.sim_R = None
        self.state.sim_meta = None
        self.state.best_fit_t = job.get("best_fit_t")
        self.state.best_fit_R = job.get("best_fit_R")
        self.state.best_fit_meta = job.get("best_fit_meta")
        self._editing_job_index = idx if job.get("status") in ("queued", "failed") else None
        self._loading_job_to_editor = True
        try:
            target_mode = "simulation" if str(job.get("type", "")).lower() in ("simulation", "simulate", "sim") else "fitting"
            self._set_mode(target_mode)
        finally:
            self._loading_job_to_editor = False
        self.btn_add_job.setText("Update job" if self._editing_job_index is not None else "Add to job list")
        self.statusBar().showMessage(f"Loaded job {idx + 1} to editor.")
        self._redraw_all()

    @staticmethod
    def _safe_filename_part(text: str) -> str:
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
        return text.strip("._") or "job"

    def _job_result_output_path(self, job: dict, suffix: str = "result") -> Path:
        if self._queue_output_dir is None:
            raise ValueError("Queue output directory is not set.")
        try:
            row = self._jobs.index(job) + 1
        except ValueError:
            row = 1
        candidate = self._job_result_file_name(job, row, suffix=suffix)
        used = getattr(self, "_queue_export_names", set())
        if candidate in used or (self._queue_output_dir / candidate).exists():
            stem = Path(candidate).stem
            ext = Path(candidate).suffix or ".mat"
            counter = 2
            while candidate in used or (self._queue_output_dir / candidate).exists():
                candidate = f"{stem}_{counter:03d}{ext}"
                counter += 1
        used.add(candidate)
        self._queue_export_names = used
        return self._queue_output_dir / candidate

    def _job_result_file_name(self, job: dict, row: int, suffix: str = "result") -> str:
        exp = job.get("experiment", {}) or {}
        raw_name = str(exp.get("file_name", "") or "job")
        stem = Path(raw_name).stem
        stem = self._clean_imported_experiment_name(stem)
        stem = Path(stem).stem
        stem = re.sub(r"(?i)(?:_?simulation)?_?result$", "", stem).strip("._ ")
        exp_stem = self._safe_filename_part(stem)
        model = self._safe_filename_part(self._job_model_label(str(job.get("model", ""))))
        suffix = self._safe_filename_part(suffix)
        return f"job_{row:03d}_{exp_stem}_{model}_{suffix}.mat"

    @staticmethod
    def _preview_curve(t, R, max_points: int = 2000) -> tuple[np.ndarray | None, np.ndarray | None]:
        if t is None or R is None:
            return None, None
        t_arr = np.asarray(t, dtype=float).reshape(-1)
        R_arr = np.asarray(R, dtype=float).reshape(-1)
        n = min(t_arr.size, R_arr.size)
        if n <= 0:
            return None, None
        t_arr = t_arr[:n]
        R_arr = R_arr[:n]
        if n <= max_points:
            return t_arr.copy(), R_arr.copy()
        idx = np.linspace(0, n - 1, max_points).astype(int)
        return t_arr[idx].copy(), R_arr[idx].copy()

    def _job_sim_spec(self, job: dict) -> tuple[_SimSpec, float, dict[str, float]]:
        exp = job["experiment"]
        phys = job["physics"]
        exp_settings = job["experiment_settings"]
        t_exp_all = np.asarray(exp.get("t", []), dtype=float)
        R_exp_all = np.asarray(exp.get("R", []), dtype=float)
        rmax_exp = find_rmax_value(t_exp_all, R_exp_all) if t_exp_all.size and R_exp_all.size else 0.0
        job_const = dict(job.get("constants", {}))
        job_const["c_long"] = float(phys.get("c_long", job_const.get("c_long", 1485.0)))
        job_const["gamma"] = float(phys.get("gamma", job_const.get("gamma", 0.056)))
        job_solver = {
            "solver_method": phys.get("solver_method", "BDF"),
            "rel_tol": float(phys.get("rel_tol", 1e-8)),
            "abs_tol": float(phys.get("abs_tol", 1e-7)),
        }
        req = float(exp_settings["Req_um"]) * 1e-6
        sim_spec = _SimSpec(
            model_key=job["model"],
            Req=req,
            NT=int(phys["NT"]),
            P_inf=float(phys["P_inf"]),
            rho=float(phys["rho"]),
            const=job_const,
            solver=job_solver,
            Rmax_exp=float(rmax_exp),
            bubble_model=phys.get("bubble_model", "Keller-Miksis"),
            plugin_entrypoint=str(job.get("solver_entrypoint", "")),
            plugin_context={
                "model_key": job["model"],
                "Req": req,
                "NT": int(phys["NT"]),
                "P_inf": float(phys["P_inf"]),
                "rho": float(phys["rho"]),
                "bubble_model": phys.get("bubble_model", "Keller-Miksis"),
                "c_long": float(job_const.get("c_long", 1485.0)),
                "gamma": float(job_const.get("gamma", 0.056)),
                "constants": job_const,
                "solver": job_solver,
                "Rmax_exp": float(rmax_exp),
            },
        )
        params = dict(job.get("_runtime_initial_values", job.get("initial_values", {})))
        tspan = float(exp_settings["tspan_us"]) * 1e-6
        return sim_spec, tspan, params

    def _make_job_sim_worker(self, job: dict) -> SimWorker:
        sim_spec, tspan, params = self._job_sim_spec(job)
        return SimWorker(
            _run_sim_spec_payload,
            {"spec": sim_spec, "params_si": params, "tspan": tspan},
            self,
        )

    def _make_job_fit_worker(self, job: dict) -> FitWorker:
        exp = job["experiment"]
        fw = job["fit_window"]
        t_exp_all = np.asarray(exp["t"], dtype=float)
        R_exp_all = np.asarray(exp["R"], dtype=float)
        mask = (t_exp_all >= float(fw["t_start_s"])) & (t_exp_all <= float(fw["t_end_s"]))
        t_windowed = t_exp_all[mask]
        R_windowed = R_exp_all[mask]

        sim_spec, _tspan, _params = self._job_sim_spec(job)
        cfg = FitConfig(
            t_exp=t_windowed,
            R_exp=R_windowed,
            make_sim=lambda params_si, tspan, spec=sim_spec: _sim_spec_call(spec, params_si, tspan),
            mp_make_sim=functools.partial(_sim_spec_call, sim_spec),
            param_names=list(job["parameters"].keys()),
        )
        return FitWorker(
            cfg,
            job["bounds_si"],
            job["fit_flags"],
            job["scales"],
            job.get("_runtime_initial_values", job["initial_values"]),
            opt_config=OptConfig(**job["optimizer"]),
            parent=self,
        )

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): MainWindow._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [MainWindow._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return value

    @staticmethod
    def _mat_to_float(mat: dict, key: str, default=None):
        if key not in mat:
            return default
        arr = np.asarray(mat[key])
        if arr.size == 0:
            return default
        try:
            return float(arr.reshape(-1)[0])
        except Exception:
            return default

    @staticmethod
    def _mat_to_string(mat: dict, key: str, default: str = "") -> str:
        if key not in mat:
            return default
        val = mat[key]
        if isinstance(val, str):
            return val
        if isinstance(val, bytes):
            return val.decode(errors="replace")
        arr = np.asarray(val)
        if arr.size == 0:
            return default
        try:
            if arr.dtype.kind in ("U", "S"):
                flat = arr.reshape(-1)
                parts = [
                    x.decode(errors="replace") if isinstance(x, bytes) else str(x)
                    for x in flat
                ]
                if flat.size > 1 and all(len(part) <= 1 for part in parts):
                    return "".join(parts)
                return parts[0]
            first = arr.reshape(-1)[0]
            if isinstance(first, bytes):
                return first.decode(errors="replace")
            return str(first)
        except Exception:
            return default

    @staticmethod
    def _none_if_nan(value):
        if value is None:
            return None
        try:
            return None if np.isnan(float(value)) else value
        except Exception:
            return value

    @staticmethod
    def _mat_array(mat: dict, key: str) -> np.ndarray:
        if key not in mat:
            return np.array([], dtype=float)
        return np.asarray(mat[key], dtype=float).reshape(-1)

    @staticmethod
    def _parse_struct_best_fit_from_mat(mat: dict) -> dict[str, dict]:
        if "struct_best_fit" not in mat:
            return {}
        flat = np.ravel(mat["struct_best_fit"])

        def _to_str(rec, field: str) -> str:
            val = getattr(rec, field)
            arr = np.asarray(val)
            if arr.size == 0:
                return ""
            item = arr.reshape(-1)[0]
            if isinstance(item, bytes):
                return item.decode(errors="replace").strip()
            return str(item).strip()

        def _to_float(rec, field: str, default: float = np.nan) -> float:
            try:
                arr = np.asarray(getattr(rec, field), dtype=float).reshape(-1)
                return float(arr[0]) if arr.size else default
            except Exception:
                return default

        out: dict[str, dict] = {}
        for rec in flat:
            try:
                name = _to_str(rec, "name")
                if not name or "MCOS" in name:
                    continue
                out[name] = {
                    "value": _to_float(rec, "value"),
                    "lb": _to_float(rec, "lb"),
                    "ub": _to_float(rec, "ub"),
                    "scale": _to_str(rec, "scale") or "lin",
                }
            except Exception:
                continue
        return out

    def _job_from_result_mat(self, path: str, mat: dict) -> dict | None:
        t_sim = self._mat_array(mat, "t_sim")
        R_sim = self._mat_array(mat, "R_sim")
        t_exp = self._mat_array(mat, "t_exp")
        R_exp = self._mat_array(mat, "R_exp")
        if t_sim.size < 3 or R_sim.size < 3 or t_exp.size < 3 or R_exp.size < 3:
            return None

        params = self._parse_struct_best_fit_from_mat(mat)
        model_key = self._mat_to_string(mat, "model_key", "") or self._get_active_model_key()
        if model_key and self._cmb_model.findText(model_key) >= 0 and model_key != self._get_active_model_key():
            self._cmb_model.setCurrentIndex(self._cmb_model.findText(model_key))

        gui_param_defaults = self._collect_parameter_defaults()
        gui_initial_values = self._get_param_si()
        gui_fit_flags, gui_scales, gui_bounds_si = self._collect_fit_setup()

        best_params = {
            name: float(meta["value"])
            for name, meta in params.items()
            if np.isfinite(float(meta.get("value", np.nan)))
        }
        param_defaults = {
            name: {
                "value_si": float(meta.get("value", np.nan)),
                "unit_index": 0,
                "fit": True,
                "lb_si": float(meta.get("lb", np.nan)),
                "ub_si": float(meta.get("ub", np.nan)),
                "scale": str(meta.get("scale", "lin") or "lin"),
            }
            for name, meta in params.items()
        }
        bounds_si = {
            name: (float(meta.get("lb", np.nan)), float(meta.get("ub", np.nan)))
            for name, meta in params.items()
        }
        scales = {name: str(meta.get("scale", "lin") or "lin") for name, meta in params.items()}
        fit_flags = {name: True for name in params}
        if not param_defaults:
            param_defaults = gui_param_defaults
        else:
            for name, fallback in gui_param_defaults.items():
                param_defaults.setdefault(name, dict(fallback))
        if not best_params:
            best_params = dict(gui_initial_values)
        else:
            for name, fallback in gui_initial_values.items():
                best_params.setdefault(name, float(fallback))
        if not bounds_si:
            bounds_si = gui_bounds_si
        else:
            for name, fallback in gui_bounds_si.items():
                bounds_si.setdefault(name, fallback)
        if not scales:
            scales = gui_scales
        else:
            for name, fallback in gui_scales.items():
                scales.setdefault(name, fallback)
        if not fit_flags:
            fit_flags = gui_fit_flags
        else:
            for name, fallback in gui_fit_flags.items():
                fit_flags.setdefault(name, fallback)

        req_m = self._none_if_nan(self._mat_to_float(mat, "Req", None))
        p_inf = self._none_if_nan(self._mat_to_float(mat, "P_inf", None))
        rho = self._none_if_nan(self._mat_to_float(mat, "rho", None))
        gamma = self._none_if_nan(self._mat_to_float(mat, "gamma", None))
        tc = self._mat_to_float(mat, "tc", 1.0)
        rmax_sim = self._mat_to_float(mat, "Rmax_sim", float(np.max(R_sim)))
        lsq_err = self._none_if_nan(self._mat_to_float(mat, "LSQErr", None))
        status = "completed"
        error = None
        if (
            bool(self._job_success_threshold_enabled)
            and lsq_err is not None
            and float(lsq_err) > float(self._job_success_lsqerr)
        ):
            status = "failed"
            error = (
                f"Imported result LSQErr {float(lsq_err):.6g} exceeds success "
                f"threshold {float(self._job_success_lsqerr):.6g}."
            )

        has_saved_window = (
            "fit_window_t_start_s" in mat
            and "fit_window_t_end_s" in mat
        )
        t_start = self._mat_to_float(mat, "fit_window_t_start_s", np.nan)
        t_end = self._mat_to_float(mat, "fit_window_t_end_s", np.nan)
        fw_mode = self._mat_to_string(mat, "fit_window_mode", "") or ""
        fw_cycles_raw = self._none_if_nan(self._mat_to_float(mat, "fit_window_cycles", None))
        if not has_saved_window or not np.isfinite(t_start) or not np.isfinite(t_end):
            model_for_window = model_key or self._get_active_model_key()
            t_start, t_end, cycles_from_gui = self._current_fit_window_seconds(
                t_exp, R_exp, model_for_window
            )
            fw_mode = "auto_cycles" if self.chk_fit_window_cycles.isChecked() else "manual"
            fw_cycles_raw = cycles_from_gui
        fw_cycles = None if fw_cycles_raw is None else int(fw_cycles_raw)
        fw_n_points = int(self._mat_to_float(
            mat,
            "fit_window_n_points",
            np.count_nonzero((t_exp >= t_start) & (t_exp <= t_end)),
        ))
        optimizer = asdict(self._opt_config)
        optimizer_json = self._mat_to_string(mat, "optimizer_json", "").strip()
        if optimizer_json:
            try:
                loaded_optimizer = json.loads(optimizer_json)
                if isinstance(loaded_optimizer, dict):
                    valid_keys = set(OptConfig.__dataclass_fields__.keys())
                    optimizer.update({
                        key: value
                        for key, value in loaded_optimizer.items()
                        if key in valid_keys
                    })
            except Exception:
                pass
        else:
            optimizer_method = self._mat_to_string(mat, "optimizer_method", "").strip()
            if optimizer_method:
                optimizer["method"] = optimizer_method
        source_name = self._mat_to_string(mat, "file_name", "") or Path(path).name
        source_name = self._clean_imported_experiment_name(source_name)
        tspan_us = float(np.max(t_sim) - np.min(t_sim)) * 1e6 if t_sim.size else 0.0
        result_kind = self._mat_to_string(mat, "imr_result_kind", "").strip().lower()
        job_type = "simulation" if result_kind == "simulation" else "fit"
        preview_t, preview_R = self._preview_curve(t_sim, R_sim)

        return {
            "version": 1,
            "type": job_type,
            "status": status,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": {
                "path": path,
                "file_name": source_name,
                "t": np.array(t_exp, dtype=float).copy(),
                "R": np.array(R_exp, dtype=float).copy(),
                "P_inf": p_inf,
                "rho": rho,
                "R_eq": req_m,
            },
            "model": model_key,
            "solver_entrypoint": self._plugin_entrypoint_for_model(model_key) if model_key else "",
            "constants": {},
            "parameters": param_defaults,
            "initial_values": dict(best_params),
            "fit_flags": fit_flags,
            "scales": scales,
            "bounds_si": bounds_si,
            "experiment_settings": {
                "Req_um": float(req_m) * 1e6 if req_m is not None else None,
                "tspan_us": tspan_us,
            },
            "fit_window": {
                "mode": fw_mode,
                "cycles": fw_cycles,
                "t_start_s": t_start,
                "t_end_s": t_end,
                "n_points": fw_n_points,
            },
            "physics": {
                "bubble_model": self._mat_to_string(mat, "bubble_model", "Keller-Miksis"),
                "P_inf": p_inf,
                "rho": rho,
                "gamma": gamma,
                "NT": int(self._mat_to_float(mat, "NT", 500)),
            },
            "optimizer": optimizer,
            "result": None,
            "lsq_err": lsq_err,
            "best_params": best_params,
            "best_fit_t": preview_t,
            "best_fit_R": preview_R,
            "best_fit_meta": {"Rmax": rmax_sim, "t_rmax": 0.0, "tc": tc},
            "archived_result_mat": None,
            "export_path": path,
            "error": error,
        }

    @staticmethod
    def _mat_bytes(data: dict) -> bytes:
        buf = BytesIO()
        savemat(buf, data)
        return buf.getvalue()

    def _job_input_mat_dict(self, job: dict) -> dict:
        exp = job.get("experiment", {})

        def col(arr):
            return np.asarray(arr, dtype=float).reshape(-1, 1)

        return {
            "t_exp": col(exp.get("t", [])),
            "R_exp": col(exp.get("R", [])),
            "P_inf": np.nan if exp.get("P_inf") is None else float(exp.get("P_inf")),
            "rho": np.nan if exp.get("rho") is None else float(exp.get("rho")),
            "R_eq": np.nan if exp.get("R_eq") is None else float(exp.get("R_eq")),
            "source_path": str(exp.get("path", "")),
            "file_name": str(exp.get("file_name", "")),
        }

    def _job_preview_mat_dict(self, job: dict) -> dict | None:
        t = job.get("best_fit_t")
        R = job.get("best_fit_R")
        if t is None or R is None:
            return None

        def col(arr):
            return np.asarray(arr, dtype=float).reshape(-1, 1)

        meta = job.get("best_fit_meta") or {}
        return {
            "best_fit_t": col(t),
            "best_fit_R": col(R),
            "Rmax": float(meta.get("Rmax", np.nan)),
            "t_rmax": float(meta.get("t_rmax", np.nan)),
            "tc": float(meta.get("tc", np.nan)),
            "LSQErr": np.nan if job.get("lsq_err") is None else float(job.get("lsq_err")),
        }

    @staticmethod
    def _job_window_lsqerr(job: dict, out: NhkvOutputs) -> tuple[float | None, int, str | None]:
        exp = job.get("experiment", {})
        if out.t_sim is None or out.R_sim is None or out.t_sim.size < 3 or out.R_sim.size < 3:
            return None, 0, "simulation output contains fewer than 3 points"
        t_exp = np.asarray(exp.get("t", []), dtype=float)
        R_exp = np.asarray(exp.get("R", []), dtype=float)
        if t_exp.size < 3 or R_exp.size < 3 or t_exp.shape[0] != R_exp.shape[0]:
            return None, 0, "experiment data is empty or invalid"

        fw = job.get("fit_window", {}) or {}
        t_start_s = float(fw.get("t_start_s", np.nan))
        t_end_s = float(fw.get("t_end_s", np.nan))
        if not np.isfinite(t_start_s) or not np.isfinite(t_end_s):
            return None, 0, "fit window is missing"
        if t_start_s > t_end_s:
            t_start_s, t_end_s = t_end_s, t_start_s

        mask = (t_exp >= t_start_s) & (t_exp <= t_end_s)
        t_windowed = t_exp[mask]
        R_windowed = R_exp[mask]
        if t_windowed.size < 3:
            return None, int(t_windowed.size), "simulation LSQErr window contains fewer than 3 points"

        t_sim = np.asarray(out.t_sim, dtype=float)
        R_sim = np.asarray(out.R_sim, dtype=float)
        covered = (t_windowed >= float(t_sim[0])) & (t_windowed <= float(t_sim[-1]))
        t_windowed = t_windowed[covered]
        R_windowed = R_windowed[covered]
        n_points = int(t_windowed.size)
        if n_points < 3:
            return None, n_points, "simulation does not cover enough points in the LSQErr window"

        try:
            R_sim_interp = np.interp(t_windowed, t_sim, R_sim)
        except Exception as exc:
            return None, n_points, f"simulation LSQErr interpolation failed: {exc}"

        err = float(np.sum(((R_windowed - R_sim_interp) * 1e6) ** 2) / max(n_points, 1))
        if not np.isfinite(err):
            return None, n_points, "simulation LSQErr is not finite"
        return err, n_points, None

    def _build_job_sim_result_export(
        self,
        job: dict,
        out: NhkvOutputs,
        lsq_err: float | None,
    ) -> dict | None:
        if out is None:
            return None

        def col(arr):
            return np.asarray(arr, dtype=float).reshape(-1, 1)

        exp = job["experiment"]
        phys = job["physics"]
        exp_settings = job["experiment_settings"]
        t_exp = np.asarray(exp.get("t", []), dtype=float)
        R_exp = np.asarray(exp.get("R", []), dtype=float)
        Rmax_exp = find_rmax_value(t_exp, R_exp) if t_exp.size and R_exp.size else np.nan
        P_inf = float(phys["P_inf"])
        rho = float(phys["rho"])
        Req = float(exp_settings["Req_um"]) * 1e-6
        Uc = float(np.sqrt(P_inf / rho)) if rho > 0 else 1.0
        tc = Req / Uc if Uc > 0 else 1.0

        export: dict = {
            "t_sim": col(out.t_sim),
            "R_sim": col(out.R_sim),
            "U_sim": col(out.U_sim),
            "P_sim": col(out.P_sim),
            "t_sim_nondim": col(out.t_sim_nondim),
            "R_sim_nondim": col(out.R_sim_nondim),
            "Rmax_sim": float(out.Rmax_sim),
            "tc": float(out.tc),
            "Uc": float(out.Uc),
            "n_damaged": int(out.n_damaged),
            "t_exp": col(t_exp),
            "R_exp": col(R_exp),
            "t_nondim_exp": col(t_exp / tc) if tc else col(t_exp),
            "R_nondim_exp": col(R_exp / Rmax_exp) if np.isfinite(Rmax_exp) and Rmax_exp else col(R_exp),
            "Rmax_exp": float(Rmax_exp),
            "P_inf": P_inf,
            "rho": rho,
            "gamma": float(phys.get("gamma", np.nan)),
            "Req": Req,
            "model_key": job["model"],
            "job_type": job.get("type", "simulation"),
            "imr_result_kind": "simulation",
            "bubble_model": phys.get("bubble_model", "Keller-Miksis"),
            "NT": int(phys.get("NT", 0)),
            "LSQErr": np.nan if lsq_err is None else float(lsq_err),
        }
        legend = str(job.get("legend", "")).strip()
        if legend:
            export["legend"] = legend
        fit_window = dict(job.get("fit_window", {}) or {})
        export["fit_window_mode"] = str(fit_window.get("mode", ""))
        export["fit_window_cycles"] = (
            np.nan if fit_window.get("cycles") is None else int(fit_window.get("cycles"))
        )
        export["fit_window_t_start_s"] = float(fit_window.get("t_start_s", np.nan))
        export["fit_window_t_end_s"] = float(fit_window.get("t_end_s", np.nan))
        export["fit_window_n_points"] = int(fit_window.get("n_points", 0))
        export["optimizer_json"] = json.dumps({}, ensure_ascii=False)
        export["optimizer_method"] = "simulation"

        names = list(job.get("parameters", {}).keys())
        dtype = np.dtype([
            ("name", "O"), ("value", "O"), ("lb", "O"),
            ("ub", "O"), ("scale", "O"), ("group", "O"),
        ])
        arr = np.empty((1, len(names)), dtype=dtype)
        params = dict(job.get("best_params", job.get("initial_values", {})))
        for i, nm in enumerate(names):
            lb, ub = job.get("bounds_si", {}).get(nm, (np.nan, np.nan))
            arr[0, i]["name"] = np.array(nm, dtype=object)
            arr[0, i]["value"] = float(params.get(nm, np.nan))
            arr[0, i]["lb"] = float(lb)
            arr[0, i]["ub"] = float(ub)
            arr[0, i]["scale"] = np.array(job.get("scales", {}).get(nm, "lin"), dtype=object)
            arr[0, i]["group"] = np.array("", dtype=object)
        export["struct_best_fit"] = arr

        return export

    def _export_job_sim_result(self, job: dict, out: NhkvOutputs, path: Path):
        export = self._build_job_sim_result_export(job, out, job.get("lsq_err"))
        if export is None:
            return
        savemat(path, export)

    def _build_job_result_export(self, job: dict, res: FitResult) -> dict | None:
        out = res.sim_out
        if out is None:
            return None

        def col(arr):
            return np.asarray(arr, dtype=float).reshape(-1, 1)

        exp = job["experiment"]
        phys = job["physics"]
        exp_settings = job["experiment_settings"]
        t_exp = np.asarray(exp["t"], dtype=float)
        R_exp = np.asarray(exp["R"], dtype=float)
        Rmax_exp = find_rmax_value(t_exp, R_exp)
        P_inf = float(phys["P_inf"])
        rho = float(phys["rho"])
        Req = float(exp_settings["Req_um"]) * 1e-6
        Uc = float(np.sqrt(P_inf / rho)) if rho > 0 else 1.0
        tc = Req / Uc if Uc > 0 else 1.0

        export: dict = {
            "t_sim": col(out.t_sim),
            "R_sim": col(out.R_sim),
            "U_sim": col(out.U_sim),
            "P_sim": col(out.P_sim),
            "t_sim_nondim": col(out.t_sim_nondim),
            "R_sim_nondim": col(out.R_sim_nondim),
            "Rmax_sim": float(out.Rmax_sim),
            "tc": float(out.tc),
            "Uc": float(out.Uc),
            "n_damaged": int(out.n_damaged),
            "t_exp": col(t_exp),
            "R_exp": col(R_exp),
            "t_nondim_exp": col(t_exp / tc),
            "R_nondim_exp": col(R_exp / Rmax_exp),
            "Rmax_exp": float(Rmax_exp),
            "P_inf": P_inf,
            "rho": rho,
            "gamma": float(phys.get("gamma", np.nan)),
            "Req": Req,
            "model_key": job["model"],
            "job_type": job.get("type", "fit"),
            "imr_result_kind": "fit",
            "LSQErr": float(res.lsq_err),
        }
        fit_window = dict(job.get("fit_window", {}) or {})
        export["fit_window_mode"] = str(fit_window.get("mode", ""))
        export["fit_window_cycles"] = (
            np.nan if fit_window.get("cycles") is None else int(fit_window.get("cycles"))
        )
        export["fit_window_t_start_s"] = float(fit_window.get("t_start_s", np.nan))
        export["fit_window_t_end_s"] = float(fit_window.get("t_end_s", np.nan))
        export["fit_window_n_points"] = int(fit_window.get("n_points", 0))
        optimizer = dict(job.get("optimizer", {}) or {})
        export["optimizer_json"] = json.dumps(self._json_safe(optimizer), ensure_ascii=False)
        export["optimizer_method"] = str(optimizer.get("method", ""))

        names = list(job["parameters"].keys())
        dtype = np.dtype([
            ("name", "O"), ("value", "O"), ("lb", "O"),
            ("ub", "O"), ("scale", "O"), ("group", "O"),
        ])
        arr = np.empty((1, len(names)), dtype=dtype)
        for i, nm in enumerate(names):
            lb, ub = job["bounds_si"].get(nm, (np.nan, np.nan))
            arr[0, i]["name"] = np.array(nm, dtype=object)
            arr[0, i]["value"] = float(res.best_params.get(nm, np.nan))
            arr[0, i]["lb"] = float(lb)
            arr[0, i]["ub"] = float(ub)
            arr[0, i]["scale"] = np.array(job["scales"].get(nm, "lin"), dtype=object)
            arr[0, i]["group"] = np.array("", dtype=object)
        export["struct_best_fit"] = arr

        return export

    def _export_job_result(self, job: dict, res: FitResult, path: Path):
        export = self._build_job_result_export(job, res)
        if export is None:
            return
        savemat(path, export)

    def _job_result_mat_bytes(self, job: dict) -> bytes | None:
        export_path = job.get("export_path")
        if export_path:
            try:
                path = Path(str(export_path))
                if path.is_file():
                    return path.read_bytes()
            except Exception:
                pass
        res = job.get("result")
        if res is not None:
            export = self._build_job_result_export(job, res)
            if export is not None:
                return self._mat_bytes(export)
        sim_out = job.get("sim_out")
        if sim_out is not None:
            export = self._build_job_sim_result_export(job, sim_out, job.get("lsq_err"))
            if export is not None:
                return self._mat_bytes(export)
        archived = job.get("archived_result_mat")
        if isinstance(archived, bytes):
            return archived
        return None

    def _job_archive_meta(self, job: dict, row: int) -> dict:
        exp = job.get("experiment", {})
        status = job.get("status", "queued")
        if status == "running":
            status = "queued"
        meta = {
            "version": int(job.get("version", 1)),
            "type": job.get("type", "fit"),
            "status": status,
            "created_at": job.get("created_at", ""),
            "experiment": {
                "path": exp.get("path", ""),
                "file_name": exp.get("file_name", ""),
                "P_inf": exp.get("P_inf"),
                "rho": exp.get("rho"),
                "R_eq": exp.get("R_eq"),
            },
            "model": job.get("model", ""),
            "solver_entrypoint": job.get("solver_entrypoint", ""),
            "constants": job.get("constants", {}),
            "parameters": job.get("parameters", {}),
            "initial_values": job.get("initial_values", {}),
            "fit_flags": job.get("fit_flags", {}),
            "scales": job.get("scales", {}),
            "bounds_si": job.get("bounds_si", {}),
            "experiment_settings": job.get("experiment_settings", {}),
            "fit_window": job.get("fit_window", {}),
            "physics": job.get("physics", {}),
            "optimizer": job.get("optimizer", {}),
            "lsq_err": job.get("lsq_err"),
            "best_params": job.get("best_params"),
            "error": job.get("error"),
            "export_path": job.get("export_path", ""),
            "input_file": f"data/job_{row:03d}_input.mat",
        }
        preview = self._job_preview_mat_dict(job)
        if preview is not None:
            meta["preview_file"] = f"previews/job_{row:03d}_preview.mat"
        if self._job_result_mat_bytes(job) is not None:
            meta["result_file"] = f"results/{self._job_result_file_name(job, row, suffix='result')}"
        return self._json_safe(meta)

    def on_export_job_queue(self):
        if self._queue_running:
            QMessageBox.warning(self, "Queue running", "Wait for the job queue to finish before exporting.")
            return
        if not self._jobs:
            QMessageBox.information(self, "No jobs", "There are no jobs to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export job queue",
            "",
            "IMR job queue (*.imrqueue);;ZIP archive (*.zip)",
        )
        if not path:
            return
        out_path = Path(path)
        if out_path.suffix.lower() not in (".imrqueue", ".zip"):
            out_path = out_path.with_suffix(".imrqueue")

        try:
            queue_meta = {
                "format": "IMR job queue",
                "version": 1,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "jobs": [],
            }
            with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as zf:
                for row, job in enumerate(self._jobs, start=1):
                    meta = self._job_archive_meta(job, row)
                    queue_meta["jobs"].append(meta)
                    zf.writestr(meta["input_file"], self._mat_bytes(self._job_input_mat_dict(job)))
                    preview_file = meta.get("preview_file")
                    preview = self._job_preview_mat_dict(job)
                    if preview_file and preview is not None:
                        zf.writestr(preview_file, self._mat_bytes(preview))
                    result_file = meta.get("result_file")
                    result_bytes = self._job_result_mat_bytes(job)
                    if result_file and result_bytes is not None:
                        zf.writestr(result_file, result_bytes)
                zf.writestr(
                    "queue.json",
                    json.dumps(self._json_safe(queue_meta), indent=2, ensure_ascii=False),
                )
            self.statusBar().showMessage(f"Exported {len(self._jobs)} job(s) to {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"{e}\n\n{traceback.format_exc()}")

    def on_export_job_queue_csv(self):
        if self._queue_running:
            QMessageBox.warning(self, "Queue running", "Wait for the job queue to finish before exporting.")
            return
        if not self._jobs:
            QMessageBox.information(self, "No jobs", "There are no jobs to export.")
            return

        out_dir = QFileDialog.getExistingDirectory(
            self,
            "Select folder for job queue CSV files",
            self._job_output_dir if self._job_output_dir and Path(self._job_output_dir).exists() else "",
        )
        if not out_dir:
            return

        groups: dict[str, list[tuple[int, dict]]] = {}
        for row, job in enumerate(self._jobs, start=1):
            model = str(job.get("model", "") or "unknown_model")
            groups.setdefault(model, []).append((row, job))

        written: list[Path] = []
        try:
            out_path = Path(out_dir)
            for model, rows in groups.items():
                param_names: list[str] = []
                seen: set[str] = set()
                for _row, job in rows:
                    best_params = job.get("best_params")
                    if isinstance(best_params, dict) and best_params:
                        names = best_params.keys()
                    else:
                        names = (job.get("parameters") or {}).keys()
                    for name in names:
                        name = str(name)
                        if name not in seen:
                            seen.add(name)
                            param_names.append(name)
                param_columns = [
                    (name, self._csv_param_label(model, name))
                    for name in param_names
                ]

                model_part = self._safe_filename_part(self._job_model_label(model))
                csv_path = out_path / f"job_queue_{model_part}.csv"
                with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "job",
                            "type",
                            "status",
                            "experiment",
                            "model",
                            "LSQErr",
                            "export_path",
                            "error",
                            *(label for _name, label in param_columns),
                        ],
                    )
                    writer.writeheader()
                    for row, job in rows:
                        best_params = job.get("best_params")
                        if not isinstance(best_params, dict):
                            best_params = {}
                        record = {
                            "job": row,
                            "type": job.get("type", ""),
                            "status": job.get("status", ""),
                            "experiment": (job.get("experiment") or {}).get("file_name", ""),
                            "model": model,
                            "LSQErr": "" if job.get("lsq_err") is None else float(job.get("lsq_err")),
                            "export_path": job.get("export_path", ""),
                            "error": job.get("error", ""),
                        }
                        for name, label in param_columns:
                            value = best_params.get(name, "")
                            try:
                                record[label] = float(value) if value != "" else ""
                            except Exception:
                                record[label] = value
                        writer.writerow(record)
                written.append(csv_path)

            self.statusBar().showMessage(
                f"Exported {len(written)} CSV file(s) to {out_path}"
            )
            QMessageBox.information(
                self,
                "CSV export completed",
                "Exported CSV file(s):\n\n" + "\n".join(str(p) for p in written),
            )
        except Exception as e:
            QMessageBox.critical(self, "CSV export failed", f"{e}\n\n{traceback.format_exc()}")

    def _job_from_archive_meta(self, meta: dict, input_mat: dict, preview_mat: dict | None,
                               result_bytes: bytes | None) -> dict:
        exp_meta = meta.get("experiment", {})
        t_exp = np.asarray(input_mat.get("t_exp", []), dtype=float).reshape(-1)
        R_exp = np.asarray(input_mat.get("R_exp", []), dtype=float).reshape(-1)
        p_inf = self._mat_to_float(input_mat, "P_inf", exp_meta.get("P_inf"))
        rho = self._mat_to_float(input_mat, "rho", exp_meta.get("rho"))
        r_eq = self._mat_to_float(input_mat, "R_eq", exp_meta.get("R_eq"))
        p_inf = self._none_if_nan(p_inf)
        rho = self._none_if_nan(rho)
        r_eq = self._none_if_nan(r_eq)

        status = meta.get("status", "queued")
        if status not in ("queued", "completed", "failed"):
            status = "queued"

        bounds = meta.get("bounds_si", {}) or {}
        bounds_si = {}
        for key, value in bounds.items():
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                bounds_si[key] = (float(value[0]), float(value[1]))
            else:
                bounds_si[key] = (np.nan, np.nan)

        job = {
            "version": int(meta.get("version", 1)),
            "type": meta.get("type", "fit"),
            "status": status,
            "created_at": meta.get("created_at", ""),
            "experiment": {
                "path": exp_meta.get("path") or self._mat_to_string(input_mat, "source_path", ""),
                "file_name": self._clean_imported_experiment_name(
                    exp_meta.get("file_name") or self._mat_to_string(input_mat, "file_name", "loaded experiment")
                ),
                "t": t_exp,
                "R": R_exp,
                "P_inf": p_inf,
                "rho": rho,
                "R_eq": r_eq,
            },
            "model": meta.get("model", ""),
            "solver_entrypoint": meta.get("solver_entrypoint", ""),
            "constants": dict(meta.get("constants", {}) or {}),
            "parameters": dict(meta.get("parameters", {}) or {}),
            "initial_values": dict(meta.get("initial_values", {}) or {}),
            "fit_flags": dict(meta.get("fit_flags", {}) or {}),
            "scales": dict(meta.get("scales", {}) or {}),
            "bounds_si": bounds_si,
            "experiment_settings": dict(meta.get("experiment_settings", {}) or {}),
            "fit_window": dict(meta.get("fit_window", {}) or {}),
            "physics": dict(meta.get("physics", {}) or {}),
            "optimizer": dict(meta.get("optimizer", {}) or {}),
            "result": None,
            "lsq_err": meta.get("lsq_err"),
            "best_params": meta.get("best_params"),
            "error": meta.get("error"),
            "export_path": meta.get("export_path", ""),
        }

        if preview_mat is not None:
            pt, pR = self._preview_curve(preview_mat.get("best_fit_t", []), preview_mat.get("best_fit_R", []))
            job["best_fit_t"] = pt
            job["best_fit_R"] = pR
            job["best_fit_meta"] = {
                "Rmax": self._mat_to_float(preview_mat, "Rmax", 1.0),
                "t_rmax": self._mat_to_float(preview_mat, "t_rmax", 0.0),
                "tc": self._mat_to_float(preview_mat, "tc", 1.0),
            }
        elif result_bytes is not None:
            try:
                result_mat = loadmat(BytesIO(result_bytes), squeeze_me=True, struct_as_record=False)
                if "t_sim" in result_mat and "R_sim" in result_mat:
                    pt, pR = self._preview_curve(result_mat["t_sim"], result_mat["R_sim"])
                    job["best_fit_t"] = pt
                    job["best_fit_R"] = pR
                    job["best_fit_meta"] = {
                        "Rmax": self._mat_to_float(result_mat, "Rmax_sim", 1.0),
                        "t_rmax": 0.0,
                        "tc": self._mat_to_float(result_mat, "tc", 1.0),
                    }
            except Exception:
                pass

        if result_bytes is not None:
            job["archived_result_mat"] = result_bytes
        return job

    def on_import_job_queue(self):
        if self._queue_running:
            QMessageBox.warning(self, "Queue running", "Wait for the job queue to finish before importing.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import job queue",
            "",
            "IMR job queue (*.imrqueue *.zip);;All files (*.*)",
        )
        if not path:
            return

        try:
            imported: list[dict] = []
            with zipfile.ZipFile(path, "r") as zf:
                queue_meta = json.loads(zf.read("queue.json").decode("utf-8"))
                if int(queue_meta.get("version", 0)) != 1:
                    raise ValueError("Unsupported job queue version.")
                job_metas = list(queue_meta.get("jobs", []))
                progress = QProgressDialog("Importing job queue...", "", 0, len(job_metas), self)
                progress.setWindowTitle("Import job queue")
                progress.setWindowModality(Qt.WindowModality.ApplicationModal)
                progress.setCancelButton(None)
                progress.setMinimumDuration(0)
                progress.setValue(0)
                try:
                    for i, meta in enumerate(job_metas, start=1):
                        exp_meta = meta.get("experiment", {})
                        label = exp_meta.get("file_name") or f"job {i}"
                        progress.setLabelText(f"Importing {i}/{len(job_metas)}: {label}")
                        progress.setValue(i - 1)
                        QApplication.processEvents()
                        input_file = meta.get("input_file")
                        if not input_file:
                            raise ValueError("Job is missing its input MAT file.")
                        input_mat = loadmat(BytesIO(zf.read(input_file)), squeeze_me=True, struct_as_record=False)
                        preview_mat = None
                        preview_file = meta.get("preview_file")
                        if preview_file:
                            preview_mat = loadmat(BytesIO(zf.read(preview_file)), squeeze_me=True, struct_as_record=False)
                        result_bytes = None
                        result_file = meta.get("result_file")
                        if result_file:
                            result_bytes = zf.read(result_file)
                        imported.append(self._job_from_archive_meta(meta, input_mat, preview_mat, result_bytes))
                        progress.setValue(i)
                        QApplication.processEvents()
                finally:
                    progress.close()

            if not imported:
                QMessageBox.information(self, "No jobs", "The selected archive does not contain any jobs.")
                return

            first_idx = len(self._jobs)
            self._jobs.extend(imported)
            self._refresh_job_table()
            self._set_mode("jobs")
            self._select_job_row(first_idx)
            self.statusBar().showMessage(f"Imported {len(imported)} job(s) from {path}")
        except Exception as e:
            QMessageBox.critical(self, "Import failed", f"{e}\n\n{traceback.format_exc()}")

    def on_run_queue(self):
        if self._queue_running:
            return
        if self._fit_worker is not None and self._fit_worker.isRunning():
            QMessageBox.warning(self, "Fitting in progress", "Wait for the current fit to finish first.")
            return
        if self._sim_worker is not None and self._sim_worker.isRunning():
            QMessageBox.warning(self, "Simulation in progress", "Wait for the current simulation to finish first.")
            return

        out_dir = self._job_output_dir.strip()
        if self._job_ask_output_dir or not out_dir or not Path(out_dir).is_dir():
            out_dir = QFileDialog.getExistingDirectory(
                self,
                "Select output folder for queue results",
                out_dir if out_dir and Path(out_dir).exists() else "",
            )
            if not out_dir:
                return
            self._job_output_dir = out_dir
            self._save_settings()

        self._queue_output_dir = Path(out_dir)
        self._queue_running = True
        self._queue_stop_after_current = False
        self._queue_previous_seed = None
        self._queue_fit_workers = {}
        self._queue_sim_workers = {}
        self._queue_export_names = set()
        effective_workers = self._effective_queue_parallel_workers()
        chain_note = " (serial because previous-best-fit chaining is enabled)" if self._job_chain_best_fit_initial else ""
        self.lbl_output.setPlainText(
            f"Starting job queue with {effective_workers} worker(s){chain_note}."
        )
        self._start_next_queue_job()

    def on_stop_queue_after_current(self):
        if self._queue_running:
            self._queue_stop_after_current = True
            self.btn_stop_jobs.setEnabled(False)
            running_workers = list(self._queue_fit_workers.values())
            if (
                self._queue_fit_worker is not None
                and self._queue_fit_worker.isRunning()
                and self._queue_fit_worker not in running_workers
            ):
                running_workers.append(self._queue_fit_worker)
            if running_workers:
                for worker in running_workers:
                    worker.request_stop()
                self.statusBar().showMessage("Stopping queue...")
            else:
                self.statusBar().showMessage("Queue will stop.")

    def _start_next_queue_job(self):
        if not self._queue_running:
            return
        if self._queue_stop_after_current:
            if self._active_queue_worker_count():
                return
            self._finish_queue("Queue stopped after current job.")
            return

        capacity = self._effective_queue_parallel_workers()
        started = False
        while self._active_queue_worker_count() < capacity:
            next_idx = None
            for i, job in enumerate(self._jobs):
                if job.get("status") == "queued":
                    next_idx = i
                    break
            if next_idx is None:
                break
            self._start_queue_job(next_idx)
            started = True

        if not started and not self._active_queue_worker_count() and not any(
            job.get("status") == "queued" for job in self._jobs
        ):
            self._finish_queue("Queue completed.")
            return

        self._update_job_buttons()

    def _active_queue_worker_count(self) -> int:
        return len(self._queue_fit_workers) + len(self._queue_sim_workers)

    def _effective_queue_parallel_workers(self) -> int:
        if self._job_chain_best_fit_initial:
            return 1
        return max(1, int(self._job_parallel_workers))

    def _start_queue_job(self, next_idx: int):
        job = self._jobs[next_idx]
        self._queue_current_index = next_idx
        job["status"] = "running"
        job["error"] = None
        job.pop("_runtime_initial_values", None)
        self._refresh_job_table()
        if self._active_queue_worker_count() == 0:
            self.tbl_jobs.selectRow(next_idx)

        if self.tbl_jobs.currentRow() == next_idx:
            self.state.sim_t = None
            self.state.sim_R = None
            self.state.sim_meta = None
            self.state.best_fit_t = None
            self.state.best_fit_R = None
            self.state.best_fit_meta = None
        job_type = str(job.get("type", "fit")).lower()
        is_sim_job = job_type in ("simulation", "simulate", "sim")
        prefix = "Running queue" if self._active_queue_worker_count() == 0 else "Starting queue"
        action = "Simulating" if is_sim_job else "Fitting"
        self.lbl_output.appendPlainText(
            f"{prefix} job {next_idx + 1}/{len(self._jobs)}: "
            f"{job['experiment']['file_name']}\n"
            f"{action} {job.get('model', '')} with "
            f"{job.get('physics', {}).get('bubble_model', 'Keller-Miksis')}"
        )
        if not is_sim_job:
            self._prepare_chained_initial_values(job, next_idx)
        self._redraw_all()

        if is_sim_job:
            worker = self._make_job_sim_worker(job)
            self._queue_sim_workers[next_idx] = worker
            worker.finished_ok.connect(lambda out, idx=next_idx, w=worker: self._on_queue_sim_ok(out, idx, w))
            worker.failed.connect(lambda msg, tb, idx=next_idx, w=worker: self._on_queue_sim_fail(msg, tb, idx, w))
        else:
            worker = self._make_job_fit_worker(job)
            self._queue_fit_workers[next_idx] = worker
            self._queue_fit_worker = worker
            worker.progress.connect(lambda prog, idx=next_idx, w=worker: self._on_queue_fit_progress(prog, idx, w))
            worker.finished_ok.connect(lambda res, idx=next_idx, w=worker: self._on_queue_fit_ok(res, idx, w))
            worker.failed.connect(lambda msg, tb, idx=next_idx, w=worker: self._on_queue_fit_fail(msg, tb, idx, w))
        worker.start()

    def _prepare_chained_initial_values(self, job: dict, idx: int):
        if not self._job_chain_best_fit_initial:
            return
        seed = self._queue_previous_seed
        if not isinstance(seed, dict):
            return
        if seed.get("type") != job.get("type") or seed.get("model") != job.get("model"):
            self.lbl_output.appendPlainText(
                "Previous best-fit seed skipped: job type or model differs from current job."
            )
            return
        prev_params = seed.get("best_params")
        if not isinstance(prev_params, dict) or not prev_params:
            return

        initial = dict(job.get("initial_values", {}))
        used = []
        for name, value in prev_params.items():
            if name in initial:
                try:
                    initial[name] = float(value)
                    used.append(name)
                except Exception:
                    pass
        if not used:
            return

        job["_runtime_initial_values"] = initial
        job["chained_initial_from_job"] = int(seed.get("index", -1)) + 1
        self.lbl_output.appendPlainText(
            f"Seeded initial values from job {int(seed.get('index', -1)) + 1}: "
            f"{', '.join(used)}"
        )

    def _remember_queue_seed_from_job(self, idx: int):
        if not self._job_chain_best_fit_initial:
            return
        if idx is None or not (0 <= idx < len(self._jobs)):
            return
        job = self._jobs[idx]
        best_params = job.get("best_params")
        if not isinstance(best_params, dict) or not best_params:
            return
        self._queue_previous_seed = {
            "index": idx,
            "type": job.get("type"),
            "model": job.get("model"),
            "best_params": dict(best_params),
        }

    def _finish_queue(self, message: str):
        self._queue_running = False
        self._queue_stop_after_current = False
        self._queue_current_index = None
        self._queue_fit_worker = None
        self._queue_fit_workers = {}
        self._queue_sim_workers = {}
        self._queue_export_names = set()
        self._queue_previous_seed = None
        self._update_job_buttons()
        self.statusBar().showMessage(message)

    def _on_queue_fit_progress(self, prog: FitProgress, idx: int | None = None, worker: FitWorker | None = None):
        if idx is None:
            idx = self._queue_current_index
        if idx is None:
            return
        job = self._jobs[idx]
        if prog.Rmax_sim is not None and prog.tc is not None:
            job["best_fit_meta"] = {"Rmax": prog.Rmax_sim, "t_rmax": 0.0, "tc": prog.tc}
        job["lsq_err"] = float(prog.best_err)
        job["best_params"] = dict(prog.best_params)
        preview_t, preview_R = self._preview_curve(prog.t_sim, prog.R_sim)
        job["best_fit_t"] = preview_t
        job["best_fit_R"] = preview_R

        selected_idx = self._selected_job_index()
        if selected_idx == idx:
            self.state.best_fit_t = preview_t
            self.state.best_fit_R = preview_R
            self.state.best_fit_meta = job.get("best_fit_meta")
        status_info = f"  [{prog.status}]" if prog.status else ""
        self.lbl_output.appendPlainText(
            f"job={idx + 1}\t|\tnfev={prog.nfev}\t|\tLSQErr={prog.best_err:.4e}"
            f"{status_info}"
        )
        target = float(job.get("optimizer", {}).get("f_tol", 0.0))
        stop_worker = worker or self._queue_fit_workers.get(idx)
        if target > 0 and prog.best_err <= target and stop_worker is not None:
            stop_worker.request_stop()
            self.lbl_output.appendPlainText(
                f"job={idx + 1}\t|\tLSQErr target reached ({target:.4e}); stopping fit"
            )
        if selected_idx == idx:
            self._redraw_all()

    def _on_queue_fit_ok(self, res: FitResult, idx: int | None = None, worker: FitWorker | None = None):
        if idx is None:
            idx = self._queue_current_index
        if idx is None:
            return
        job = self._jobs[idx]
        job["lsq_err"] = float(res.lsq_err)
        has_valid_sim = (
            res.sim_out is not None
            and res.t_sim is not None
            and res.R_sim is not None
            and np.asarray(res.t_sim).size >= 3
            and np.asarray(res.R_sim).size >= 3
            and np.isfinite(float(res.lsq_err))
            and float(res.lsq_err) < 1e10
        )
        threshold_enabled = bool(self._job_success_threshold_enabled)
        threshold_value = float(self._job_success_lsqerr)
        if not has_valid_sim:
            job["status"] = "failed"
            job["error"] = (
                "Optimizer did not produce a valid simulation result. "
                "This usually means every evaluated simulation failed or did not cover the fitting window."
            )
        elif threshold_enabled and float(res.lsq_err) > threshold_value:
            job["status"] = "failed"
            job["error"] = (
                f"Final LSQErr {res.lsq_err:.6g} exceeds success threshold "
                f"{threshold_value:.6g}."
            )
        else:
            job["status"] = "completed"
        job["result"] = res
        job["best_params"] = dict(res.best_params)
        if res.t_sim is not None and res.R_sim is not None:
            preview_t, preview_R = self._preview_curve(res.t_sim, res.R_sim)
            job["best_fit_t"] = preview_t
            job["best_fit_R"] = preview_R
            job["best_fit_meta"] = {
                "Rmax": res.Rmax_sim or 1.0,
                "t_rmax": 0.0,
                "tc": res.tc or 1.0,
            }

        if self._queue_output_dir is not None and has_valid_sim:
            out_path = self._job_result_output_path(job, "result")
            try:
                self._export_job_result(job, res, out_path)
                job["export_path"] = str(out_path)
                job["result"] = None
            except Exception as e:
                job["status"] = "failed"
                job["error"] = f"Export failed: {e}"

        done_worker = worker or self._queue_fit_workers.get(idx)
        if done_worker is not None:
            done_worker.wait(5000)
        self._queue_fit_workers.pop(idx, None)
        self._queue_fit_worker = next(iter(self._queue_fit_workers.values()), None)
        self._remember_queue_seed_from_job(idx)
        self._refresh_job_table()
        if self._queue_stop_after_current:
            if not self._active_queue_worker_count():
                self._finish_queue("Queue stopped.")
            else:
                self._update_job_buttons()
            return
        QTimer.singleShot(0, self._start_next_queue_job)

    def _on_queue_fit_fail(self, msg: str, tb: str, idx: int | None = None, worker: FitWorker | None = None):
        if idx is None:
            idx = self._queue_current_index
        if idx is None:
            return
        job = self._jobs[idx]
        job["status"] = "failed"
        job["error"] = f"{msg}\n\n{tb}"
        done_worker = worker or self._queue_fit_workers.get(idx)
        if done_worker is not None:
            done_worker.wait(5000)
        self._queue_fit_workers.pop(idx, None)
        self._queue_fit_worker = next(iter(self._queue_fit_workers.values()), None)
        self._remember_queue_seed_from_job(idx)
        self._refresh_job_table()
        if self._queue_stop_after_current:
            if not self._active_queue_worker_count():
                self._finish_queue("Queue stopped.")
            else:
                self._update_job_buttons()
            return
        QTimer.singleShot(0, self._start_next_queue_job)

    def _on_queue_sim_ok(self, out: NhkvOutputs, idx: int | None = None, worker: SimWorker | None = None):
        if idx is None:
            idx = self._queue_current_index
        if idx is None:
            return
        job = self._jobs[idx]
        lsq_err, n_lsq, lsq_msg = self._job_window_lsqerr(job, out)
        job["lsq_err"] = lsq_err
        job["lsq_n_points"] = n_lsq
        job["sim_out"] = out
        job["best_params"] = dict(job.get("initial_values", {}))
        preview_t, preview_R = self._preview_curve(out.t_sim, out.R_sim)
        job["best_fit_t"] = preview_t
        job["best_fit_R"] = preview_R
        job["best_fit_meta"] = {
            "Rmax": float(out.Rmax_sim) if out.Rmax_sim is not None else 1.0,
            "t_rmax": 0.0,
            "tc": float(out.tc) if out.tc is not None else 1.0,
        }

        threshold_enabled = bool(self._job_success_threshold_enabled)
        threshold_value = float(self._job_success_lsqerr)
        has_experiment = np.asarray(job.get("experiment", {}).get("t", []), dtype=float).size >= 3
        if lsq_err is None and not has_experiment:
            job["status"] = "completed"
            job["error"] = None
        elif lsq_err is None:
            job["status"] = "failed"
            job["error"] = lsq_msg or "Simulation LSQErr could not be computed."
        elif threshold_enabled and float(lsq_err) > threshold_value:
            job["status"] = "failed"
            job["error"] = (
                f"Simulation LSQErr {lsq_err:.6g} exceeds success threshold "
                f"{threshold_value:.6g}."
            )
        else:
            job["status"] = "completed"
            job["error"] = None

        if self._queue_output_dir is not None:
            out_path = self._job_result_output_path(job, "result")
            try:
                self._export_job_sim_result(job, out, out_path)
                job["export_path"] = str(out_path)
                job["sim_out"] = None
            except Exception as e:
                job["status"] = "failed"
                job["error"] = f"Export failed: {e}"

        selected_idx = self._selected_job_index()
        if selected_idx == idx:
            self.state.best_fit_t = job.get("best_fit_t")
            self.state.best_fit_R = job.get("best_fit_R")
            self.state.best_fit_meta = job.get("best_fit_meta")
        if lsq_err is None:
            self.lbl_output.appendPlainText(
                f"job={idx + 1}\t|\tsimulation completed\t|\tLSQErr not computed: {lsq_msg or 'unknown'}"
            )
        else:
            self.lbl_output.appendPlainText(
                f"job={idx + 1}\t|\tsimulation completed\t|\tLSQErr={lsq_err:.4e} over {n_lsq} point(s)"
            )

        done_worker = worker or self._queue_sim_workers.get(idx)
        if done_worker is not None:
            done_worker.wait(5000)
        self._queue_sim_workers.pop(idx, None)
        self._refresh_job_table()
        if selected_idx == idx:
            self._redraw_all()
        if self._queue_stop_after_current:
            if not self._active_queue_worker_count():
                self._finish_queue("Queue stopped.")
            else:
                self._update_job_buttons()
            return
        QTimer.singleShot(0, self._start_next_queue_job)

    def _on_queue_sim_fail(self, msg: str, tb: str, idx: int | None = None, worker: SimWorker | None = None):
        if idx is None:
            idx = self._queue_current_index
        if idx is None:
            return
        job = self._jobs[idx]
        job["status"] = "failed"
        job["error"] = f"{msg}\n\n{tb}"
        done_worker = worker or self._queue_sim_workers.get(idx)
        if done_worker is not None:
            done_worker.wait(5000)
        self._queue_sim_workers.pop(idx, None)
        self._refresh_job_table()
        if self._queue_stop_after_current:
            if not self._active_queue_worker_count():
                self._finish_queue("Queue stopped.")
            else:
                self._update_job_buttons()
            return
        QTimer.singleShot(0, self._start_next_queue_job)

    # =====================================================================
    # simulation
    # =====================================================================

    def on_simulate(self):
        if self._queue_running:
            QMessageBox.warning(self, "Queue running", "Wait for the job queue to finish first.")
            return
        if self._sim_worker is not None and self._sim_worker.isRunning():
            return
        if self._fit_worker is not None and self._fit_worker.isRunning():
            QMessageBox.warning(
                self, "Fitting in progress",
                "Wait for fitting to finish or stop it before simulating.",
            )
            return

        self.state.best_fit_t = None
        self.state.best_fit_R = None
        self.state.best_fit_meta = None

        self.btn_primary_action.setEnabled(False)
        model_key = self._get_active_model_key()
        bubble_model = self._bubble_model
        self.statusBar().showMessage(f"Simulating {model_key} ({bubble_model}, LIC)...")

        if self.chk_fit_window_cycles.isChecked() and self.state.exp_t is not None:
            self._apply_fit_window_cycles()

        params = self._get_param_si()
        inp = self._build_sim_inputs(params)
        sim_fn = self._get_simulate_fn()

        self._sim_worker = SimWorker(sim_fn, inp, self)

        dlg = QProgressDialog(f"Simulating {model_key} ({bubble_model}, LIC)...", "", 0, 0, self)
        dlg.setWindowTitle("Simulation in progress")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setRange(0, 0)
        dlg.show()
        self._sim_dialog = dlg

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._sim_start_time = time.time()
        self._sim_timer.start()

        def _on_ok(out):
            QApplication.restoreOverrideCursor()
            self._sim_timer.stop()
            elapsed = None
            if self._sim_start_time is not None:
                elapsed = time.time() - self._sim_start_time
                self._last_sim_duration = elapsed
                self._sim_start_time = None
            if self._sim_dialog is not None:
                self._sim_dialog.close()
                self._sim_dialog = None
            self.btn_primary_action.setEnabled(self.state.mode == "simulation")

            self.state.sim_t = out.t_sim
            self.state.sim_R = out.R_sim
            self.state.sim_meta = {"Rmax": out.Rmax_sim, "t_rmax": 0.0, "tc": out.tc}
            self.state.sim_out = out

            if self._curve_view_active():
                self._add_curve_to_view(
                    curve_type="simulation",
                    t=out.t_sim,
                    R=out.R_sim,
                    legend=f"{model_key} simulation {len(self._view_curves) + 1}",
                    meta=self.state.sim_meta,
                )

            self._redraw_all()

            used_pinf = float(self.spin_P_inf.value())
            used_rho = float(self.spin_rho.value())
            _C = 22  # fixed column width for each data field
            _hdr = f"Simulation ({model_key}, {bubble_model}):"
            _ind = " " * len(_hdr)
            _r = f"Rmax={out.Rmax_sim*1e6:.3f} µm"
            _t = f"tc={out.tc*1e6:.3f} µs"
            _u = f"Uc={out.Uc:.3f} m/s"
            _p = f"P_inf={used_pinf:.1f} Pa"
            _rh = f"rho={used_rho:.1f} kg/m³"
            _rq = f"Req={float(self.spin_Req_um.value()):.3f} µm"
            lines = [
                f"{_hdr}  {_r:<{_C}}| {_t:<{_C}}| {_u}",
                f"{_ind}  {_p:<{_C}}| {_rh:<{_C}}| {_rq}",
            ]
            lsq_err, n_lsq, lsq_msg = self._simulation_window_lsqerr(out)
            if lsq_err is not None:
                t0_us = float(self.spin_t_fit_start.value())
                t1_us = float(self.spin_t_fit_end.value())
                if t0_us > t1_us:
                    t0_us, t1_us = t1_us, t0_us
                lines.append(
                    f"{_ind}  LSQErr={lsq_err:.4e} over {n_lsq} point(s) "
                    f"[{t0_us:.3f}, {t1_us:.3f}] us"
                )
            elif lsq_msg:
                lines.append(f"{_ind}  LSQErr not computed: {lsq_msg}")
            self.lbl_output.setPlainText("\n".join(lines))
            self.statusBar().showMessage("Simulation completed")

        def _on_fail(msg: str, tb: str):
            QApplication.restoreOverrideCursor()
            self._sim_timer.stop()
            if self._sim_start_time is not None:
                self._last_sim_duration = time.time() - self._sim_start_time
                self._sim_start_time = None
            if self._sim_dialog is not None:
                self._sim_dialog.close()
                self._sim_dialog = None
            self.btn_primary_action.setEnabled(self.state.mode == "simulation")
            QMessageBox.critical(self, "Simulation failed", f"{msg}\n\n{tb}")
            self.statusBar().showMessage("Simulation failed")

        self._sim_worker.finished_ok.connect(_on_ok)
        self._sim_worker.failed.connect(_on_fail)
        self._sim_worker.start()

    def _update_sim_progress(self):
        if self._sim_dialog is None or self._sim_start_time is None:
            return
        elapsed = time.time() - self._sim_start_time
        if self._last_sim_duration is not None and self._last_sim_duration > 1e-3:
            eta = max(0.0, self._last_sim_duration - elapsed)
        else:
            eta = None
        model_key = self._get_active_model_key()
        msg = f"Simulating {model_key} ({self._bubble_model}, LIC)...\nElapsed {self._sec_to_hms(elapsed)}"
        if eta is not None:
            msg += f"  |  ETA {self._sec_to_hms(eta)}"
        self._sim_dialog.setLabelText(msg)

    def _simulation_window_lsqerr(self, out: NhkvOutputs) -> tuple[float | None, int, str | None]:
        """Compute the same windowed LSQErr used by fitting for a simulation."""
        if self.state.exp_t is None or self.state.exp_R is None:
            return None, 0, None
        if out.t_sim is None or out.R_sim is None or out.t_sim.size < 3 or out.R_sim.size < 3:
            return None, 0, "simulation output contains fewer than 3 points"

        t_exp = np.asarray(self.state.exp_t, dtype=float)
        R_exp = np.asarray(self.state.exp_R, dtype=float)
        t_start_s, t_end_s, _cycles = self._current_fit_window_seconds(
            t_exp, R_exp, self._get_active_model_key()
        )
        mask = (t_exp >= t_start_s) & (t_exp <= t_end_s)
        t_windowed = t_exp[mask]
        R_windowed = R_exp[mask]
        n_points = int(t_windowed.size)
        if n_points < 3:
            return None, n_points, "simulation LSQErr window contains fewer than 3 points"

        t_sim = np.asarray(out.t_sim, dtype=float)
        R_sim = np.asarray(out.R_sim, dtype=float)
        covered = (t_windowed >= float(t_sim[0])) & (t_windowed <= float(t_sim[-1]))
        t_windowed = t_windowed[covered]
        R_windowed = R_windowed[covered]
        n_points = int(t_windowed.size)
        if n_points < 3:
            return None, n_points, "simulation does not cover enough points in the LSQErr window"

        try:
            R_sim_interp = np.interp(
                t_windowed,
                t_sim,
                R_sim,
            )
        except Exception as exc:
            return None, n_points, f"simulation LSQErr interpolation failed: {exc}"

        err = float(np.sum(((R_windowed - R_sim_interp) * 1e6) ** 2) / max(n_points, 1))
        if not np.isfinite(err):
            return None, n_points, "simulation LSQErr is not finite"
        return err, n_points, None

    # =====================================================================
    # fitting
    # =====================================================================

    def on_fit(self):
        if self._queue_running:
            QMessageBox.warning(self, "Queue running", "Wait for the job queue to finish first.")
            return
        if self._fit_worker is not None and self._fit_worker.isRunning():
            return

        if self.state.exp_t is None or self.state.exp_R is None:
            QMessageBox.warning(self, "No data", "Please load experiment data before fitting.")
            return
        param_names = list(self._param_rows.keys())
        fit_flags: dict[str, bool] = {}
        scales: dict[str, str] = {}
        bounds_si: dict[str, tuple[float, float]] = {}

        for name in param_names:
            fw = self._fit_widgets.get(name)
            if not fw:
                continue
            fit_flags[name] = fw["chk_fit"].isChecked()
            scales[name] = fw["cmb_scale"].currentText()
            factor = self._get_unit_factor(name)
            lb = float(fw["spin_lb"].value()) * factor
            ub = float(fw["spin_ub"].value()) * factor
            if lb > ub:
                lb, ub = ub, lb
            bounds_si[name] = (lb, ub)

        if not any(fit_flags.values()):
            QMessageBox.warning(
                self, "No parameters selected",
                "Enable the Fit checkbox for at least one parameter.",
            )
            return

        initial_values = self._get_param_si()
        if not self._confirm_initial_values_within_bounds(
            initial_values, bounds_si, fit_flags
        ):
            return

        if self.chk_fit_window_cycles.isChecked():
            self._apply_fit_window_cycles()

        t_exp = self.state.exp_t
        R_exp = self.state.exp_R
        t_start_s, t_end_s, _cycles = self._current_fit_window_seconds(
            t_exp, R_exp, self._get_active_model_key()
        )
        mask = (t_exp >= t_start_s) & (t_exp <= t_end_s)
        t_windowed = t_exp[mask]
        R_windowed = R_exp[mask]

        if t_windowed.size < 3:
            QMessageBox.warning(
                self, "Too few points",
                "The fit window contains fewer than 3 data points.",
            )
            return

        _rmax_exp = (
            find_rmax_value(self.state.exp_t, self.state.exp_R)
            if self.state.exp_t is not None and self.state.exp_R is not None
            else 0.0
        )
        sim_spec = _SimSpec(
            model_key=self._get_active_model_key(),
            Req=float(self.spin_Req_um.value()) * 1e-6,
            NT=int(self.spin_NT.value()),
            P_inf=float(self.spin_P_inf.value()),
            rho=float(self.spin_rho.value()),
            const=dict(self._model_constants),
            solver=self._get_solver_settings(),
            Rmax_exp=_rmax_exp,
            bubble_model=self._bubble_model,
            plugin_entrypoint=self._plugin_entrypoint_for_model(self._get_active_model_key()),
            plugin_context=self._plugin_context(
                model_key=self._get_active_model_key(),
                Req=float(self.spin_Req_um.value()) * 1e-6,
                NT=int(self.spin_NT.value()),
                P_inf=float(self.spin_P_inf.value()),
                rho=float(self.spin_rho.value()),
                const=dict(self._model_constants),
                solver=self._get_solver_settings(),
                Rmax_exp=_rmax_exp,
            ),
        )
        cfg = FitConfig(
            t_exp=t_windowed,
            R_exp=R_windowed,
            make_sim=self._make_sim_for_fit,
            mp_make_sim=functools.partial(_sim_spec_call, sim_spec),
            param_names=param_names,
        )

        self.state.sim_t = None
        self.state.sim_R = None
        self.state.sim_meta = None
        self.state.best_fit_t = None
        self.state.best_fit_R = None
        self.state.best_fit_meta = None
        model_key = self._get_active_model_key()
        bubble_model = self._bubble_model
        self.lbl_output.setPlainText(f"Fitting {model_key} with {bubble_model}")
        self._redraw_all()

        self._fit_worker = FitWorker(
            cfg, bounds_si, fit_flags, scales, initial_values,
            opt_config=self._opt_config, parent=self,
        )

        dlg = QProgressDialog(f"Fitting {model_key} with {bubble_model} to experiment...", "Stop", 0, 0, self)
        dlg.setWindowTitle("Fitting in progress")
        dlg.setWindowModality(Qt.NonModal)
        dlg.setMinimumDuration(0)
        dlg.setRange(0, 0)
        dlg.show()
        self._fit_dialog = dlg

        self._fit_start_time = time.time()
        self._fit_timer.start()
        self.btn_primary_action.setEnabled(False)

        self._fit_worker.progress.connect(self._on_fit_progress)
        self._fit_worker.finished_ok.connect(self._on_fit_ok)
        self._fit_worker.failed.connect(self._on_fit_fail)
        dlg.canceled.connect(self._on_fit_stop_requested)

        self._fit_worker.start()

    def _on_fit_stop_requested(self):
        if self._fit_worker is not None:
            self._fit_worker.request_stop()
        self.statusBar().showMessage(
            "Stopping fit... (waiting for current evaluation to finish)"
        )
        if self._fit_dialog is not None:
            QTimer.singleShot(0, self._show_stopping_dialog)

    def _show_stopping_dialog(self):
        if self._fit_dialog is not None:
            self._fit_dialog.setRange(0, 0)
            self._fit_dialog.setLabelText(
                "Stopping... (waiting for current evaluation to finish)"
            )
            self._fit_dialog.setCancelButton(None)
            self._fit_dialog.show()

    # ---- fit signal handlers ---------------------------------------------

    def _on_fit_progress(self, prog: FitProgress):
        self.state.best_fit_t = prog.t_sim
        self.state.best_fit_R = prog.R_sim
        if prog.Rmax_sim is not None and prog.tc is not None:
            self.state.best_fit_meta = {
                "Rmax": prog.Rmax_sim, "t_rmax": 0.0, "tc": prog.tc,
            }

        bp = prog.best_params
        for name, row in self._param_rows.items():
            if name in bp:
                factor = self._get_unit_factor(name)
                row["spin"].blockSignals(True)
                row["spin"].setValue(bp[name] / factor)
                row["spin"].blockSignals(False)

        elapsed = ""
        if self._fit_start_time is not None:
            elapsed = f"\t|\telapsed {self._sec_to_hms(time.time() - self._fit_start_time)}"
        step_info = ""
        if prog.step_size is not None:
            step_info = f"\t|\tstep={prog.step_size:.3e}"
        status_info = ""
        if prog.status:
            status_info = f"  [{prog.status}]"
        self.lbl_output.appendPlainText(
            f"nfev={prog.nfev}\t|\tLSQErr={prog.best_err:.4e}"
            f"{step_info}{status_info}{elapsed}"
        )

        self._redraw_all()

        if self._fit_dialog is not None and self._fit_start_time is not None:
            el = time.time() - self._fit_start_time
            model_key = self._get_active_model_key()
            self._fit_dialog.setLabelText(
                f"Fitting {model_key}...\n"
                f"Elapsed {self._sec_to_hms(el)}\n"
                f"Best LSQErr: {prog.best_err:.4e}  |  nfev: {prog.nfev}"
            )

    def _on_fit_ok(self, res: FitResult):
        self._fit_timer.stop()
        elapsed = None
        if self._fit_start_time is not None:
            elapsed = time.time() - self._fit_start_time
            self._fit_start_time = None
        if self._fit_dialog is not None:
            self._fit_dialog.close()
            self._fit_dialog = None
        was_stopped = False
        if self._fit_worker is not None:
            was_stopped = self._fit_worker._stop_requested
            self._fit_worker.wait(5000)
            self._fit_worker = None

        self.btn_primary_action.setEnabled(self.state.mode == "fitting")

        bp = res.best_params
        for name, row in self._param_rows.items():
            if name in bp:
                factor = self._get_unit_factor(name)
                row["spin"].blockSignals(True)
                row["spin"].setValue(bp[name] / factor)
                row["spin"].blockSignals(False)

        if res.t_sim is not None and res.R_sim is not None:
            self.state.sim_t = res.t_sim
            self.state.sim_R = res.R_sim
            self.state.sim_meta = {
                "Rmax": res.Rmax_sim or 1.0,
                "t_rmax": 0.0,
                "tc": res.tc or 1.0,
            }
            self.state.sim_out = res.sim_out

        self.state.best_fit_t = None
        self.state.best_fit_R = None
        self.state.best_fit_meta = None
        self._redraw_all()

        label = "Fit stopped" if was_stopped else "Fit completed"
        extra = f", elapsed {self._sec_to_hms(elapsed)}" if elapsed else ""
        self.lbl_output.appendPlainText(
            f"--- {label}{extra}\t|\tnfev={res.nfev}\t|\tLSQErr={res.lsq_err:.4e}"
        )
        self.statusBar().showMessage(label)

    def _on_fit_fail(self, msg: str, tb: str):
        self._fit_timer.stop()
        if self._fit_start_time is not None:
            self._fit_start_time = None
        if self._fit_dialog is not None:
            self._fit_dialog.close()
            self._fit_dialog = None
        if self._fit_worker is not None:
            self._fit_worker.wait(5000)
            self._fit_worker = None
        self.btn_primary_action.setEnabled(self.state.mode == "fitting")
        QMessageBox.critical(self, "Fitting failed", f"{msg}\n\n{tb}")
        self.statusBar().showMessage("Fitting failed")

    def _update_fit_progress(self):
        if self._fit_dialog is None or self._fit_start_time is None:
            return
        elapsed = time.time() - self._fit_start_time
        model_key = self._get_active_model_key()
        msg = f"Fitting {model_key} to experiment...\nElapsed {self._sec_to_hms(elapsed)}"
        self._fit_dialog.setLabelText(msg)

    # =====================================================================
    # parameter save / load
    # =====================================================================

    def on_save_params(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save parameters", "", "MAT files (*.mat)")
        if not path:
            return

        names = list(self._param_rows.keys())
        params = self._get_param_si()

        n = len(names)
        dtype = np.dtype([
            ("name", "O"), ("value", "O"), ("lb", "O"),
            ("ub", "O"), ("scale", "O"), ("group", "O"),
        ])
        arr = np.empty((1, n), dtype=dtype)
        for i, nm in enumerate(names):
            fw = self._fit_widgets.get(nm, {})
            factor = self._get_unit_factor(nm)
            lb = float(fw["spin_lb"].value()) * factor if fw else params[nm] / 10
            ub = float(fw["spin_ub"].value()) * factor if fw else params[nm] * 10
            sc = fw["cmb_scale"].currentText() if fw else "lin"

            arr[0, i]["name"] = np.array(nm, dtype=object)
            arr[0, i]["value"] = float(params[nm])
            arr[0, i]["lb"] = float(lb)
            arr[0, i]["ub"] = float(ub)
            arr[0, i]["scale"] = np.array(sc, dtype=object)
            arr[0, i]["group"] = np.array("", dtype=object)

        try:
            savemat(path, {"struct_best_fit": arr})
            self.statusBar().showMessage(f"Parameters saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{e}\n\n{traceback.format_exc()}")

    def on_export_result(self):
        out = self.state.sim_out
        if out is None:
            QMessageBox.information(
                self, "No result",
                "Run a simulation or fitting first before exporting."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export result", "", "MAT files (*.mat)"
        )
        if not path:
            return

        try:
            def col(arr):
                return np.asarray(arr, dtype=float).reshape(-1, 1)

            export: dict = {}

            # --- simulation ---
            export["t_sim"]        = col(out.t_sim)
            export["R_sim"]        = col(out.R_sim)
            export["U_sim"]        = col(out.U_sim)
            export["P_sim"]        = col(out.P_sim)
            export["t_sim_nondim"] = col(out.t_sim_nondim)
            export["R_sim_nondim"] = col(out.R_sim_nondim)
            export["Rmax_sim"]     = float(out.Rmax_sim)
            export["tc"]           = float(out.tc)
            export["Uc"]           = float(out.Uc)
            export["n_damaged"]    = int(out.n_damaged)

            # --- experimental (if loaded) ---
            if self.state.exp_t is not None and self.state.exp_R is not None:
                t_exp = self.state.exp_t
                R_exp = self.state.exp_R
                Rmax_exp = find_rmax_value(t_exp, R_exp)
                P_inf_gui = float(self.spin_P_inf.value())
                rho_gui   = float(self.spin_rho.value())
                R_eq_gui  = float(self.spin_Req_um.value()) * 1e-6
                Uc_gui    = float(np.sqrt(P_inf_gui / rho_gui)) if rho_gui > 0 else 1.0
                tc_gui    = R_eq_gui / Uc_gui if Uc_gui > 0 else 1.0
                export["t_exp"]        = col(t_exp)
                export["R_exp"]        = col(R_exp)
                export["t_nondim_exp"] = col(t_exp / tc_gui)
                export["R_nondim_exp"] = col(R_exp / Rmax_exp)
                export["Rmax_exp"]     = float(Rmax_exp)
                t0, t1, cycles = self._current_fit_window_seconds(
                    t_exp, R_exp, self._get_active_model_key()
                )
                export["fit_window_mode"] = (
                    "auto_cycles" if self.chk_fit_window_cycles.isChecked() else "manual"
                )
                export["fit_window_cycles"] = np.nan if cycles is None else int(cycles)
                export["fit_window_t_start_s"] = float(t0)
                export["fit_window_t_end_s"] = float(t1)
                export["fit_window_n_points"] = int(np.count_nonzero((t_exp >= t0) & (t_exp <= t1)))
                import_meta = getattr(self.state, "import_metadata", None)
                if import_meta:
                    safe_meta = self._json_safe(import_meta)
                    export["struct_import"] = {
                        str(key): value for key, value in safe_meta.items()
                    }

            # --- parameters (same struct format as Save parameters) ---
            names  = list(self._param_rows.keys())
            params = self._get_param_si()
            n = len(names)
            dtype = np.dtype([
                ("name", "O"), ("value", "O"), ("lb", "O"),
                ("ub", "O"), ("scale", "O"), ("group", "O"),
            ])
            arr = np.empty((1, n), dtype=dtype)
            for i, nm in enumerate(names):
                fw     = self._fit_widgets.get(nm, {})
                factor = self._get_unit_factor(nm)
                lb = float(fw["spin_lb"].value()) * factor if fw else params[nm] / 10
                ub = float(fw["spin_ub"].value()) * factor if fw else params[nm] * 10
                sc = fw["cmb_scale"].currentText() if fw else "lin"
                arr[0, i]["name"]  = np.array(nm, dtype=object)
                arr[0, i]["value"] = float(params[nm])
                arr[0, i]["lb"]    = float(lb)
                arr[0, i]["ub"]    = float(ub)
                arr[0, i]["scale"] = np.array(sc, dtype=object)
                arr[0, i]["group"] = np.array("", dtype=object)
            export["struct_best_fit"] = arr

            # --- metadata scalars ---
            export["P_inf"]     = float(self.spin_P_inf.value())
            export["rho"]       = float(self.spin_rho.value())
            export["gamma"]     = float(self.spin_gamma.value())
            export["Req"]       = float(self.spin_Req_um.value()) * 1e-6
            export["model_key"] = self._get_active_model_key()
            optimizer = asdict(self._opt_config)
            export["optimizer_json"] = json.dumps(self._json_safe(optimizer), ensure_ascii=False)
            export["optimizer_method"] = str(self._opt_config.method)

            savemat(path, export)
            self.statusBar().showMessage(f"Result exported to {path}")

        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"{e}\n\n{traceback.format_exc()}")

    def on_load_params(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load parameters", "", "MAT files (*.mat)"
        )
        if not path:
            return
        self._load_params_from_path(path)

    def _load_params_from_path(self, path: str):
        try:
            # --- load MAT file; try mat73 first (handles MATLAB string type),
            #     fall back to scipy for older v5 files with char arrays ---
            src: dict[str, dict] = {}

            def _build_src_mat73(path: str):
                """Parse struct_best_fit via mat73 (HDF5 / v7.3 MAT files)."""
                m = _mat73.loadmat(path)
                if "struct_best_fit" not in m:
                    raise ValueError("MAT file does not contain 'struct_best_fit'")
                sb = m["struct_best_fit"]
                # mat73 returns a struct array as a dict of lists:
                #   {'name': [...], 'value': [...], 'lb': [...], ...}
                names  = sb.get("name",  [])
                vals   = sb.get("value", [])
                lbs    = sb.get("lb",    [])
                ubs    = sb.get("ub",    [])
                scales = sb.get("scale", [])
                result = {}
                for i, raw_name in enumerate(names):
                    try:
                        name = str(raw_name).strip()
                        result[name] = dict(
                            value=float(vals[i])   if i < len(vals)   else 0.0,
                            lb   =float(lbs[i])    if i < len(lbs)    else 0.0,
                            ub   =float(ubs[i])    if i < len(ubs)    else 1e10,
                            scale=str(scales[i]).strip() if i < len(scales) else "lin",
                        )
                    except Exception:
                        pass
                return result

            def _build_src_scipy(path: str):
                """Parse struct_best_fit via scipy (v5 MAT files).

                Handles two cases:
                  - char-array names: decoded directly.
                  - MCOS string objects (modern MATLAB): names cannot be decoded;
                    falls back to positional layout matching by struct entry count.
                """
                # Known struct layouts ordered by parameter position (from JSON).
                # Used when MATLAB 'string' type blocks name decoding.
                _LAYOUTS: dict[int, list[str]] = {
                    2:  ["G", "mu"],                                        # NHKV (Rmax)
                    3:  ["U0", "G", "mu"],                                  # NHKV
                    7:  ["U0", "GA", "alpha", "GB", "beta", "mu", "lambda_Y"],
                    11: ["U0", "GA1", "GA2", "alpha1", "alpha2",
                         "GB1", "GB2", "beta1", "beta2", "mu", "lambda_Y"],
                }

                m = loadmat(path, squeeze_me=True, struct_as_record=False)
                if "struct_best_fit" not in m:
                    raise ValueError("MAT file does not contain 'struct_best_fit'")
                sb   = m["struct_best_fit"]
                flat = np.ravel(sb)

                def _to_str(rec, field):
                    val = getattr(rec, field)
                    arr = np.asarray(val)
                    return str(arr.item()).strip() if arr.ndim == 0 else str(arr.squeeze()).strip()

                def _is_mcos(s: str) -> bool:
                    return "MCOS" in s or s.startswith("(b'")

                def _to_float(rec, field):
                    val = getattr(rec, field)
                    arr = np.asarray(val).astype(float).ravel()
                    if arr.size == 0:
                        raise ValueError(f"Field '{field}' is empty")
                    return float(arr[0])

                result = {}
                for rec in flat:
                    try:
                        name = _to_str(rec, "name")
                        if _is_mcos(name):
                            continue        # MCOS string — handled below
                        scale_raw = _to_str(rec, "scale")
                        result[name] = dict(
                            value=_to_float(rec, "value"),
                            lb   =_to_float(rec, "lb"),
                            ub   =_to_float(rec, "ub"),
                            scale=scale_raw if not _is_mcos(scale_raw) else "lin",
                        )
                    except Exception:
                        pass

                # Positional fallback when MCOS blocked all name decoding
                if not result:
                    layout = _LAYOUTS.get(len(flat))
                    if layout:
                        for i, rec in enumerate(flat):
                            if i >= len(layout):
                                break
                            try:
                                name = layout[i]
                                scale_raw = _to_str(rec, "scale")
                                result[name] = dict(
                                    value=_to_float(rec, "value"),
                                    lb   =_to_float(rec, "lb"),
                                    ub   =_to_float(rec, "ub"),
                                    scale=scale_raw if not _is_mcos(scale_raw) else "lin",
                                )
                            except Exception:
                                pass

                return result

            # Try mat73 first; if it fails or produces MCOS garbage, try scipy
            loaded = False
            if _HAS_MAT73:
                try:
                    src = _build_src_mat73(path)
                    # Reject if any name looks like an MCOS object repr
                    if src and not any("MCOS" in k for k in src):
                        loaded = True
                except Exception:
                    pass
            if not loaded:
                src = _build_src_scipy(path)

            known_names = set(self._param_rows.keys())

            def _stem(name: str) -> str:
                """Strip trailing '1' or '2' to get the canonical parameter stem.
                e.g. 'GA1' → 'GA', 'alpha2' → 'alpha', 'GA' → 'GA'."""
                return name[:-1] if name and name[-1] in "12" else name

            def _best_match(target: str) -> dict | None:
                """Find the best source record for *target*.

                Matching order:
                  1. Exact name  (GA1 → GA1)
                  2. Stem-based  (GA → GA1, alpha → alpha1, GA1 → GA)
                  3. Zero-substitution: if best candidate has value 0,
                     fall through to the next suffix variant
                     (alpha1=0 → try alpha2, per user request).

                For targets ending in '2' only exact or '2'-suffixed sources
                are accepted, so loading a 1-term file into GMOD2 does NOT
                overwrite the second-branch parameters.
                """
                # 1) exact match
                if target in src:
                    return src[target]

                tgt_stem = _stem(target)
                tgt_sfx = target[len(tgt_stem):]   # "", "1", or "2"

                # '2'-suffix targets: only accept exact (handled above) or a
                # same-stem '2'-suffixed source.  Never map bare/1-term params
                # onto the second branch.
                if tgt_sfx == "2":
                    d = src.get(tgt_stem + "2")
                    return d  # None if not present → spinbox keeps current value

                # bare or '1'-suffix target: gather stem-matching candidates
                # priority: '1'-suffix → bare-name → '2'-suffix
                _PRIO = {"1": 0, "": 1, "2": 2}
                candidates: list[tuple[int, dict]] = []
                for sname, d in src.items():
                    s_stem = _stem(sname)
                    s_sfx = sname[len(s_stem):]
                    if s_stem == tgt_stem:
                        candidates.append((_PRIO.get(s_sfx, 9), d))

                if not candidates:
                    return None

                candidates.sort(key=lambda x: x[0])

                # Within each priority level prefer non-zero values;
                # if the best priority is all zeros, fall through to next level
                # (this is the alpha1=0 → alpha2 substitution).
                current_prio = None
                same_level: list[dict] = []
                for prio, d in candidates:
                    if prio != current_prio:
                        non_zero = [x for x in same_level if x["value"] != 0.0]
                        if non_zero:
                            return non_zero[0]
                        current_prio = prio
                        same_level = []
                    same_level.append(d)
                # flush last level
                non_zero = [x for x in same_level if x["value"] != 0.0]
                if non_zero:
                    return non_zero[0]
                # all candidates are zero — return first anyway
                return candidates[0][1]

            values: dict[str, float] = {}
            param_bounds: dict[str, dict] = {}

            for target in known_names:
                d = _best_match(target)
                if d is not None:
                    values[target] = d["value"]
                    param_bounds[target] = {"lb": d["lb"], "ub": d["ub"], "scale": d["scale"]}

            for name, val in values.items():
                row = self._param_rows.get(name)
                if row:
                    factor = self._get_unit_factor(name)
                    row["spin"].setValue(val / factor)

            for name, meta in param_bounds.items():
                fw = self._fit_widgets.get(name)
                if fw:
                    factor = self._get_unit_factor(name)
                    fw["spin_lb"].setValue(float(meta["lb"]) / factor)
                    fw["spin_ub"].setValue(float(meta["ub"]) / factor)
                    fw["cmb_scale"].setCurrentText(meta.get("scale", "lin"))

            if values:
                self.state.param_bounds = param_bounds
                self.statusBar().showMessage(f"Parameters loaded from {path}")
            else:
                src_names = sorted(src.keys())
                tgt_names = sorted(known_names)
                loader = "mat73" if (loaded and _HAS_MAT73) else "scipy"
                raise ValueError(
                    f"Could not map struct_best_fit to current model parameters "
                    f"(loaded via {loader}).\n\n"
                    f"MAT file names:      {src_names}\n\n"
                    f"Current model needs: {tgt_names}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"{e}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run():
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    run()
