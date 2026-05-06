from __future__ import annotations

import functools
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
try:
    import mat73 as _mat73
    _HAS_MAT73 = True
except ImportError:
    _HAS_MAT73 = False
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QActionGroup, QColor, QPen, QValidator
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
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
from imr_gui.ui.mpl_canvas import MplCanvas
from imr_gui.constitutive import (
    load_nhkv_model,
    ConstitutiveParameter, ConstitutiveModel, AVAILABLE_MODELS,
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


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


@dataclass
class AppState:
    exp_t: np.ndarray | None = None
    exp_R: np.ndarray | None = None
    exp_path: str | None = None
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
    else:  # GMOD2
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMR Fitting GUI (beta 1.1)")
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
        self._opt_config: OptConfig = OptConfig()
        self._job_success_threshold_enabled: bool = True
        self._job_success_lsqerr: float = 10.0
        self._job_output_dir: str = ""
        self._job_ask_output_dir: bool = True
        self._job_chain_best_fit_initial: bool = False
        self._queue_previous_seed: dict | None = None

        # model state
        self._current_model: ConstitutiveModel | None = None
        self._model_constants: dict = {}
        self._param_rows: dict[str, dict] = {}
        self._fit_widgets: dict[str, dict] = {}
        self._saved_ui_defaults: dict = {}
        self._jobs: list[dict] = []
        self._editing_job_index: int | None = None

        self._build_menu()
        self._build_ui()
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
        self._act_batch_add_jobs = file_menu.addAction("Batch add experiments as jobs...")
        self._act_batch_add_jobs.triggered.connect(self.on_batch_add_experiments_as_jobs)

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
        for key in AVAILABLE_MODELS:
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
        self.spin_fit_window_cycles.setRange(1, 20)
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

        self._update_view_buttons()

        # populate initial model
        self._on_model_changed(self._cmb_model.currentText())
        self._cmb_model.currentTextChanged.connect(self._on_model_changed)

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

            warn_lbl = QLabel(
                "⚠  Warning: do not modify unless you know what you are doing!"
            )
            warn_lbl.setStyleSheet(
                "QLabel { color: #cc6600; font-weight: bold; padding: 4px; "
                "border: 1px solid #cc6600; border-radius: 3px; }"
            )
            warn_lbl.setWordWrap(True)
            lay.addWidget(warn_lbl)

            form = QFormLayout()
            form.addRow("P_inf (Pa):", self.spin_P_inf)
            form.addRow("rho (kg/m³):", self.spin_rho)
            form.addRow("c_long (m/s):", self.spin_c_long)
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

        self._physics_dlg.exec()

    def _toggle_advanced_solver(self, checked: bool):
        self._adv_solver_widget.setVisible(checked)
        self._adv_toggle.setText(
            "▾ Advanced Solver Settings" if checked
            else "▸ Advanced Solver Settings"
        )

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

        def _browse_output_dir():
            start_dir = le_output_dir.text().strip()
            if not start_dir or not Path(start_dir).exists():
                start_dir = ""
            path = QFileDialog.getExistingDirectory(self, "Select default job output folder", start_dir)
            if path:
                le_output_dir.setText(path)

        btn_browse.clicked.connect(_browse_output_dir)

        grid.addWidget(chk_threshold, 0, 0)
        grid.addWidget(QLabel("Success LSQErr threshold:"), 0, 1)
        grid.addWidget(spin_threshold, 0, 2)
        grid.addWidget(chk_ask_dir, 1, 0)
        grid.addWidget(QLabel("Default output folder:"), 1, 1)
        grid.addLayout(output_row, 1, 2)
        grid.addWidget(chk_chain_initial, 2, 0, 1, 3)
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
        c.ps_debug_log = self._chk_ps_debug.isChecked()
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
        if "parameters" in ui and ui.get("model") in AVAILABLE_MODELS:
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

    def _parameter_defaults_for_model(self, model_key: str) -> dict:
        ui = self._normalise_ui_defaults(self._saved_ui_defaults)
        model_data = ui.get("models", {}).get(model_key, {})
        params = model_data.get("parameters", {})
        return params if isinstance(params, dict) else {}

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
        if model_key in AVAILABLE_MODELS:
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
        loader = AVAILABLE_MODELS.get(model_key)
        if not loader:
            return
        model = loader()
        self._current_model = model
        self._model_constants = model.constants
        self._param_box.setTitle(f"Parameters ({model.display_name})")
        self._rebuild_param_panel(model)
        self._apply_parameter_defaults(self._parameter_defaults_for_model(model_key))
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

        self.state.mode = mode
        is_fitting = mode == "fitting"
        is_jobs = mode == "jobs"
        if not is_fitting:
            self._editing_job_index = None

        for fw in self._fit_widgets.values():
            fw["row_widget"].setEnabled(is_fitting)
            fw["cmb_scale"].setEnabled(is_fitting)

        self._fit_window_widget.setVisible(is_fitting)
        if hasattr(self, "_left_stack"):
            self._left_stack.setCurrentIndex(1 if is_jobs else 0)
        self.btn_primary_action.setText("Fit" if is_fitting else "Simulate")
        self.btn_primary_action.setEnabled(not is_jobs)
        self.btn_add_job.setEnabled(is_fitting)
        self.btn_add_job.setText("Update job" if is_fitting and self._editing_job_index is not None else "Add to job list")
        self._act_load_exp.setEnabled(not is_jobs)
        self._act_load_params.setEnabled(not is_jobs)
        self._act_save_params.setEnabled(not is_jobs)
        self._act_export_result.setEnabled(not is_jobs)
        self._act_batch_add_jobs.setEnabled(is_fitting)

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

    def _build_sim_inputs(self, params: dict[str, float]):
        """Build solver inputs from GUI params + constants for the current model."""
        key = self._get_active_model_key()
        const = dict(self._model_constants)
        const["c_long"] = float(self.spin_c_long.value())
        Req = float(self.spin_Req_um.value()) * 1e-6
        tspan = float(self.spin_tspan_us.value()) * 1e-6
        NT = int(self.spin_NT.value())
        solver = self._get_solver_settings()
        P_inf = float(self.spin_P_inf.value())
        rho = float(self.spin_rho.value())

        bm = self._bubble_model
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
        Req = float(self.spin_Req_um.value()) * 1e-6
        NT = int(self.spin_NT.value())
        solver = self._get_solver_settings()
        P_inf = float(self.spin_P_inf.value())
        rho = float(self.spin_rho.value())

        bm = self._bubble_model
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
        if self.state.mode == "fitting":
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
        if self.state.mode == "fitting":
            self._redraw_all()

    def _on_fit_window_auto_changed(self, checked: bool):
        self.spin_t_fit_start.setEnabled(not checked)
        self.spin_t_fit_end.setEnabled(not checked)
        if checked:
            self._apply_fit_window_cycles()
        elif self.state.mode == "fitting":
            self._redraw_all()

    def _on_fit_window_cycles_changed(self, _value: int):
        if self.chk_fit_window_cycles.isChecked():
            self._apply_fit_window_cycles()

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
            t_exp = np.asarray(preview_exp_t, dtype=float)
            R_exp = np.asarray(preview_exp_R, dtype=float)
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
            self.canvas.set_drag_limits(float(t_plot[0]), float(t_plot[-1]))

        if (
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

        if self.state.mode == "fitting" and self.state.exp_t is not None:
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
            exp = load_experiment_mat(path)
            self.state.exp_t = exp.t
            self.state.exp_R = exp.R
            self.state.exp_path = exp.source_path
            self.state.P_inf = exp.P_inf
            self.state.rho = exp.rho
            self.state.R_eq = exp.R_eq

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
                QMessageBox.critical(
                    self, "Load failed", f"{e}\n\n{traceback.format_exc()}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"{e}\n\n{traceback.format_exc()}")

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
        editable_statuses = ("queued", "failed")
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
        if self.chk_fit_window_cycles.isChecked():
            cycles = int(self.spin_fit_window_cycles.value())
            i0, i1, _found = self._fit_window_cycle_indices_for(exp_t, exp_R, cycles, model_key)
            t_start_s = float(exp_t[i0])
            t_end_s = float(exp_t[i1])
            window_mode = "auto_cycles"
        else:
            t_start_s = float(self.spin_t_fit_start.value()) * 1e-6
            t_end_s = float(self.spin_t_fit_end.value()) * 1e-6
            cycles = int(self.spin_fit_window_cycles.value())
            window_mode = "manual"
        if t_start_s > t_end_s:
            t_start_s, t_end_s = t_end_s, t_start_s

        mask = (exp_t >= t_start_s) & (exp_t <= t_end_s)
        n_points = int(np.count_nonzero(mask))
        if n_points < 3:
            return None, f"Fit window contains fewer than 3 points ({n_points})."

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
            },
            "model": model_key,
            "constants": dict(self._model_constants),
            "parameters": self._collect_parameter_defaults(),
            "initial_values": self._get_param_si(),
            "fit_flags": dict(fit_flags),
            "scales": dict(scales),
            "bounds_si": dict(bounds_si),
            "experiment_settings": {
                "Req_um": float(self.spin_Req_um.value()),
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

        if self.chk_fit_window_cycles.isChecked():
            self._apply_fit_window_cycles()

        exp_path = self.state.exp_path or ""
        exp_name = Path(exp_path).name if exp_path else "loaded experiment"
        job, error = self._build_fit_job_snapshot_from_data(
            exp_t=self.state.exp_t,
            exp_R=self.state.exp_R,
            exp_path=exp_path,
            exp_name=exp_name,
            exp_P_inf=self.state.P_inf,
            exp_rho=self.state.rho,
            exp_R_eq=self.state.R_eq,
            fit_flags=fit_flags,
            scales=scales,
            bounds_si=bounds_si,
        )
        if job is None:
            QMessageBox.warning(self, "Cannot add job", error or "Could not build fit job.")
        return job

    def on_add_job(self):
        if self.state.mode != "fitting":
            QMessageBox.information(
                self, "Fitting jobs only",
                "The first job queue version supports fitting jobs only. Switch to Fitting mode first.",
            )
            return
        job = self._build_fit_job_snapshot()
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
        if self.state.mode != "fitting":
            QMessageBox.information(
                self,
                "Fitting mode required",
                "Switch to Fitting mode before batch-adding experiments as jobs.",
            )
            return

        fit_flags, scales, bounds_si = self._collect_fit_setup()
        if not any(fit_flags.values()):
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
            for i, path in enumerate(paths, start=1):
                progress.setLabelText(f"Loading {i}/{len(paths)}: {Path(path).name}")
                progress.setValue(i - 1)
                QApplication.processEvents()
                try:
                    exp = load_experiment_mat(path)
                    exp_R_eq = exp.R_eq
                    if exp_R_eq is None and exp.R.size > 0:
                        exp_R_eq = float(np.mean(exp.R[-min(20, exp.R.size):]))
                    job, error = self._build_fit_job_snapshot_from_data(
                        exp_t=exp.t,
                        exp_R=exp.R,
                        exp_path=exp.source_path,
                        exp_name=Path(exp.source_path).name,
                        exp_P_inf=exp.P_inf,
                        exp_rho=exp.rho,
                        exp_R_eq=exp_R_eq,
                        fit_flags=fit_flags,
                        scales=scales,
                        bounds_si=bounds_si,
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

    def on_remove_selected_job(self):
        idx = self._selected_job_index()
        if idx is None:
            return
        if self._jobs[idx].get("status") not in ("queued", "failed"):
            QMessageBox.information(self, "Cannot remove", "Only queued or failed jobs can be removed.")
            return
        del self._jobs[idx]
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
        self._set_mode("fitting")
        self.btn_add_job.setText("Update job" if self._editing_job_index is not None else "Add to job list")
        self.statusBar().showMessage(f"Loaded job {idx + 1} to editor.")
        self._redraw_all()

    @staticmethod
    def _safe_filename_part(text: str) -> str:
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
        return text.strip("._") or "job"

    def _make_job_fit_worker(self, job: dict) -> FitWorker:
        exp = job["experiment"]
        fw = job["fit_window"]
        t_exp_all = np.asarray(exp["t"], dtype=float)
        R_exp_all = np.asarray(exp["R"], dtype=float)
        mask = (t_exp_all >= float(fw["t_start_s"])) & (t_exp_all <= float(fw["t_end_s"]))
        t_windowed = t_exp_all[mask]
        R_windowed = R_exp_all[mask]

        phys = job["physics"]
        exp_settings = job["experiment_settings"]
        rmax_exp = find_rmax_value(t_exp_all, R_exp_all)
        sim_spec = _SimSpec(
            model_key=job["model"],
            Req=float(exp_settings["Req_um"]) * 1e-6,
            NT=int(phys["NT"]),
            P_inf=float(phys["P_inf"]),
            rho=float(phys["rho"]),
            const=dict(job.get("constants", {})),
            solver={
                "solver_method": phys.get("solver_method", "BDF"),
                "rel_tol": float(phys.get("rel_tol", 1e-8)),
                "abs_tol": float(phys.get("abs_tol", 1e-7)),
            },
            Rmax_exp=float(rmax_exp),
            bubble_model=phys.get("bubble_model", "Keller-Miksis"),
        )
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
        arr = np.asarray(val)
        if arr.size == 0:
            return default
        try:
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
            "Req": Req,
            "model_key": job["model"],
            "LSQErr": float(res.lsq_err),
        }

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
        res = job.get("result")
        if res is not None:
            export = self._build_job_result_export(job, res)
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
            meta["result_file"] = f"results/job_{row:03d}_result.mat"
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
                "file_name": exp_meta.get("file_name") or self._mat_to_string(input_mat, "file_name", "loaded experiment"),
                "t": t_exp,
                "R": R_exp,
                "P_inf": p_inf,
                "rho": rho,
                "R_eq": r_eq,
            },
            "model": meta.get("model", ""),
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
            job["best_fit_t"] = np.asarray(preview_mat.get("best_fit_t", []), dtype=float).reshape(-1)
            job["best_fit_R"] = np.asarray(preview_mat.get("best_fit_R", []), dtype=float).reshape(-1)
            job["best_fit_meta"] = {
                "Rmax": self._mat_to_float(preview_mat, "Rmax", 1.0),
                "t_rmax": self._mat_to_float(preview_mat, "t_rmax", 0.0),
                "tc": self._mat_to_float(preview_mat, "tc", 1.0),
            }
        elif result_bytes is not None:
            try:
                result_mat = loadmat(BytesIO(result_bytes), squeeze_me=True, struct_as_record=False)
                if "t_sim" in result_mat and "R_sim" in result_mat:
                    job["best_fit_t"] = np.asarray(result_mat["t_sim"], dtype=float).reshape(-1)
                    job["best_fit_R"] = np.asarray(result_mat["R_sim"], dtype=float).reshape(-1)
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
        self._start_next_queue_job()

    def on_stop_queue_after_current(self):
        if self._queue_running:
            self._queue_stop_after_current = True
            self.btn_stop_jobs.setEnabled(False)
            if self._queue_fit_worker is not None and self._queue_fit_worker.isRunning():
                self._queue_fit_worker.request_stop()
                self.statusBar().showMessage("Stopping current queue job...")
            else:
                self.statusBar().showMessage("Queue will stop after the current job.")

    def _start_next_queue_job(self):
        if self._queue_stop_after_current:
            self._finish_queue("Queue stopped after current job.")
            return

        next_idx = None
        for i, job in enumerate(self._jobs):
            if job.get("status") == "queued":
                next_idx = i
                break

        if next_idx is None:
            self._finish_queue("Queue completed.")
            return

        self._queue_current_index = next_idx
        job = self._jobs[next_idx]
        job["status"] = "running"
        job["error"] = None
        job.pop("_runtime_initial_values", None)
        self._refresh_job_table()
        self.tbl_jobs.selectRow(next_idx)

        self.state.sim_t = None
        self.state.sim_R = None
        self.state.sim_meta = None
        self.state.best_fit_t = None
        self.state.best_fit_R = None
        self.state.best_fit_meta = None
        self.lbl_output.setPlainText(
            f"Running queue job {next_idx + 1}/{len(self._jobs)}: "
            f"{job['experiment']['file_name']}"
        )
        self._prepare_chained_initial_values(job, next_idx)
        self._redraw_all()

        self._queue_fit_worker = self._make_job_fit_worker(job)
        self._queue_fit_worker.progress.connect(self._on_queue_fit_progress)
        self._queue_fit_worker.finished_ok.connect(self._on_queue_fit_ok)
        self._queue_fit_worker.failed.connect(self._on_queue_fit_fail)
        self._queue_fit_worker.start()
        self._update_job_buttons()

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
        self._queue_previous_seed = None
        self._update_job_buttons()
        self.statusBar().showMessage(message)

    def _on_queue_fit_progress(self, prog: FitProgress):
        idx = self._queue_current_index
        if idx is None:
            return
        job = self._jobs[idx]
        if prog.Rmax_sim is not None and prog.tc is not None:
            job["best_fit_meta"] = {"Rmax": prog.Rmax_sim, "t_rmax": 0.0, "tc": prog.tc}
        job["lsq_err"] = float(prog.best_err)
        job["best_params"] = dict(prog.best_params)
        job["best_fit_t"] = np.array(prog.t_sim, dtype=float).copy() if prog.t_sim is not None else None
        job["best_fit_R"] = np.array(prog.R_sim, dtype=float).copy() if prog.R_sim is not None else None

        self.state.best_fit_t = prog.t_sim
        self.state.best_fit_R = prog.R_sim
        self.state.best_fit_meta = job.get("best_fit_meta")
        status_info = f"  [{prog.status}]" if prog.status else ""
        self.lbl_output.appendPlainText(
            f"job={idx + 1}\t|\tnfev={prog.nfev}\t|\tLSQErr={prog.best_err:.4e}"
            f"{status_info}"
        )
        target = float(job.get("optimizer", {}).get("f_tol", 0.0))
        if target > 0 and prog.best_err <= target and self._queue_fit_worker is not None:
            self._queue_fit_worker.request_stop()
            self.lbl_output.appendPlainText(
                f"job={idx + 1}\t|\tLSQErr target reached ({target:.4e}); stopping fit"
            )
        self._redraw_all()

    def _on_queue_fit_ok(self, res: FitResult):
        idx = self._queue_current_index
        if idx is None:
            return
        job = self._jobs[idx]
        job["lsq_err"] = float(res.lsq_err)
        threshold_enabled = bool(self._job_success_threshold_enabled)
        threshold_value = float(self._job_success_lsqerr)
        if threshold_enabled and float(res.lsq_err) > threshold_value:
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
            job["best_fit_t"] = np.array(res.t_sim, dtype=float).copy()
            job["best_fit_R"] = np.array(res.R_sim, dtype=float).copy()
            job["best_fit_meta"] = {
                "Rmax": res.Rmax_sim or 1.0,
                "t_rmax": 0.0,
                "tc": res.tc or 1.0,
            }

        if self._queue_output_dir is not None:
            exp_stem = self._safe_filename_part(Path(job["experiment"]["file_name"]).stem)
            model = self._safe_filename_part(self._job_model_label(job["model"]))
            out_path = self._queue_output_dir / f"job_{idx + 1:03d}_{exp_stem}_{model}_result.mat"
            try:
                self._export_job_result(job, res, out_path)
                job["export_path"] = str(out_path)
            except Exception as e:
                job["status"] = "failed"
                job["error"] = f"Export failed: {e}"

        if self._queue_fit_worker is not None:
            self._queue_fit_worker.wait(5000)
        self._queue_fit_worker = None
        self._remember_queue_seed_from_job(idx)
        self._refresh_job_table()
        QTimer.singleShot(0, self._start_next_queue_job)

    def _on_queue_fit_fail(self, msg: str, tb: str):
        idx = self._queue_current_index
        if idx is None:
            return
        job = self._jobs[idx]
        job["status"] = "failed"
        job["error"] = f"{msg}\n\n{tb}"
        if self._queue_fit_worker is not None:
            self._queue_fit_worker.wait(5000)
        self._queue_fit_worker = None
        self._remember_queue_seed_from_job(idx)
        self._refresh_job_table()
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
        self.statusBar().showMessage(f"Simulating {model_key} (LIC)...")

        params = self._get_param_si()
        inp = self._build_sim_inputs(params)
        sim_fn = self._get_simulate_fn()

        self._sim_worker = SimWorker(sim_fn, inp, self)

        dlg = QProgressDialog(f"Simulating {model_key} (LIC)...", "", 0, 0, self)
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

            self._redraw_all()

            used_pinf = float(self.spin_P_inf.value())
            used_rho = float(self.spin_rho.value())
            _C = 22  # fixed column width for each data field
            _hdr = f"Simulation ({model_key}):"
            _ind = " " * len(_hdr)
            _r = f"Rmax={out.Rmax_sim*1e6:.3f} µm"
            _t = f"tc={out.tc*1e6:.3f} µs"
            _u = f"Uc={out.Uc:.3f} m/s"
            _p = f"P_inf={used_pinf:.1f} Pa"
            _rh = f"rho={used_rho:.1f} kg/m³"
            _rq = f"Req={float(self.spin_Req_um.value()):.3f} µm"
            self.lbl_output.setPlainText(
                f"{_hdr}  {_r:<{_C}}| {_t:<{_C}}| {_u}\n"
                f"{_ind}  {_p:<{_C}}| {_rh:<{_C}}| {_rq}"
            )
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
        msg = f"Simulating {model_key} (LIC)...\nElapsed {self._sec_to_hms(elapsed)}"
        if eta is not None:
            msg += f"  |  ETA {self._sec_to_hms(eta)}"
        self._sim_dialog.setLabelText(msg)

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

        if self.chk_fit_window_cycles.isChecked():
            self._apply_fit_window_cycles()

        t_start_s = float(self.spin_t_fit_start.value()) * 1e-6
        t_end_s = float(self.spin_t_fit_end.value()) * 1e-6
        if t_start_s > t_end_s:
            t_start_s, t_end_s = t_end_s, t_start_s

        t_exp = self.state.exp_t
        R_exp = self.state.exp_R
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
        self.lbl_output.clear()
        self._redraw_all()

        self._fit_worker = FitWorker(
            cfg, bounds_si, fit_flags, scales, initial_values,
            opt_config=self._opt_config, parent=self,
        )

        model_key = self._get_active_model_key()
        dlg = QProgressDialog(f"Fitting {model_key} to experiment...", "Stop", 0, 0, self)
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
            export["Req"]       = float(self.spin_Req_um.value()) * 1e-6
            export["model_key"] = self._get_active_model_key()

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
