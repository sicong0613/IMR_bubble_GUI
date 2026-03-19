from __future__ import annotations

import functools
import traceback
from dataclasses import dataclass
import time
import math
import re

import numpy as np
from scipy.io import loadmat, savemat
try:
    import mat73 as _mat73
    _HAS_MAT73 = True
except ImportError:
    _HAS_MAT73 = False
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QActionGroup, QValidator
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
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
    QSplitter,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from imr_gui.imr import NhkvInputs, NhkvOutputs, simulate_nhkv_lic
from imr_gui.imr import GMODInputs, simulate_gmod_lic
from imr_gui.imr import GMOD1Inputs, simulate_gmod1_lic
from imr_gui.io import load_experiment_mat
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
    P_inf: float | None = None
    rho: float | None = None
    R_eq: float | None = None
    param_bounds: dict[str, dict] | None = None
    mode: str = "simulation"  # "simulation" | "fitting"
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


def _sim_spec_call(spec: _SimSpec, params_si: dict, tspan: float):
    """Module-level dispatch function — picklable for ProcessPoolExecutor workers."""
    key = spec.model_key
    if key == "NHKV":
        const_kw = {k: v for k, v in spec.const.items()
                    if k in NhkvInputs.__dataclass_fields__}
        return simulate_nhkv_lic(NhkvInputs(
            U0=params_si["U0"], G=params_si["G"], mu=params_si["mu"],
            Req=spec.Req, tspan=tspan, NT=spec.NT,
            P_inf=spec.P_inf, rho=spec.rho, **spec.solver, **const_kw,
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
            damage_index=params_si.get("damage_index", 0),
            Req=spec.Req, tspan=tspan, NT=spec.NT,
            P_inf=spec.P_inf, rho=spec.rho, **spec.solver, **const_kw,
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
            damage_index=params_si.get("damage_index", 0),
            Req=spec.Req, tspan=tspan, NT=spec.NT,
            P_inf=spec.P_inf, rho=spec.rho, **spec.solver, **const_kw,
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
        self.setWindowTitle("IMR Fitting GUI (prototype)")
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
        self._opt_config: OptConfig = OptConfig()

        # model state
        self._current_model: ConstitutiveModel | None = None
        self._model_constants: dict = {}
        self._param_rows: dict[str, dict] = {}
        self._fit_widgets: dict[str, dict] = {}

        self._build_menu()
        self._build_ui()
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")
        self._set_mode("simulation")

    # =====================================================================
    # menu
    # =====================================================================

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        act_load_exp = file_menu.addAction("Load experiment data (.mat)")
        act_load_exp.triggered.connect(self.on_load_experiment)
        file_menu.addSeparator()
        act_load_params = file_menu.addAction("Load parameters (MAT)...")
        act_load_params.triggered.connect(self.on_load_params)
        act_save_params = file_menu.addAction("Save parameters (MAT)...")
        act_save_params.triggered.connect(self.on_save_params)

        module_menu = self.menuBar().addMenu("Module")
        self._act_sim = module_menu.addAction("Simulation")
        self._act_fit_mode = module_menu.addAction("Fitting")
        self._act_sim.setCheckable(True)
        self._act_fit_mode.setCheckable(True)
        self._act_sim.setChecked(True)
        ag = QActionGroup(self)
        ag.addAction(self._act_sim)
        ag.addAction(self._act_fit_mode)
        ag.setExclusive(True)
        self._act_sim.triggered.connect(lambda: self._set_mode("simulation"))
        self._act_fit_mode.triggered.connect(lambda: self._set_mode("fitting"))

        physics_menu = self.menuBar().addMenu("Physics")
        physics_menu.addAction("Bubble dynamics: Keller-Miksis (fixed)").setEnabled(False)
        act_phys_settings = physics_menu.addAction("Physics Settings...")
        act_phys_settings.triggered.connect(self._show_physics_settings)

        settings_menu = self.menuBar().addMenu("Settings")
        act_opt_settings = settings_menu.addAction("Optimizer Settings...")
        act_opt_settings.triggered.connect(self._show_optimizer_settings)

    # =====================================================================
    # UI build
    # =====================================================================

    def _build_ui(self):
        root = QWidget(self)
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Horizontal, root)

        # ---- left panel: parameters -------------------------------------
        left = QWidget()
        left_layout = QVBoxLayout(left)

        # ---- Model selector row ----
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._cmb_model = QComboBox()
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

        self.btn_simulate = QPushButton("Simulate")
        self.btn_simulate.clicked.connect(self.on_simulate)
        left_layout.addWidget(self.btn_simulate)

        self.btn_fit = QPushButton("Fit")
        self.btn_fit.clicked.connect(self.on_fit)
        left_layout.addWidget(self.btn_fit)

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

        self.spin_NT = _NoWheelSpinBox()
        self.spin_NT.setRange(50, 2000)
        self.spin_NT.setDecimals(0)
        self.spin_NT.setSingleStep(50)
        self.spin_NT.setValue(500)
        self.spin_NT.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self._cmb_solver = QComboBox()
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
            btn_box.rejected.connect(dlg.accept)
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

    def _build_optimizer_settings_dlg(self):
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QFormLayout,
            QComboBox, QSpinBox, QCheckBox,
            QGroupBox, QStackedWidget, QLabel, QDialogButtonBox,
            QWidget,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("Optimizer Settings")
        dlg.setMinimumWidth(420)
        root = QVBoxLayout(dlg)

        # ── Algorithm ──────────────────────────────────────────────────
        form_top = QFormLayout()
        self._cmb_opt_method = QComboBox()
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
            "Initial mesh size relative to each parameter's optimizer-space range."
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
        self._opt_stack.addWidget(pg_ps)        # index 2

        # page 3 – Differential Evolution
        pg_de = QWidget()
        fde = QFormLayout(pg_de)
        self._cmb_de_strategy = QComboBox()
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
        self._opt_stack.addWidget(pg_de)        # index 3

        # page 4 – CMA-ES (only present if cma installed)
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
            "Differential Evolution": 3,
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
        btn_box.accepted.connect(lambda: (self._opt_dlg_save(), dlg.accept()))
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
        self._set_mode(self.state.mode)  # refresh fit-control visibility
        # GMOD models require much tighter ODE tolerances than NHKV.
        if model_key in ("GMOD1", "GMOD2"):
            self.le_rtol.setText("1e-9")
            self.le_atol.setText("1e-9")
        else:
            self.le_rtol.setText("1e-8")
            self.le_atol.setText("1e-7")

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

        for p in model.parameters:
            self._add_param_row(p, lay)

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
            unit_combo = QComboBox()
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

        cmb_scale = QComboBox()
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

    def _set_mode(self, mode: str):
        if self._fit_worker is not None and self._fit_worker.isRunning():
            if not self._fit_worker._stop_requested:
                QMessageBox.warning(
                    self, "Fitting in progress",
                    "Cannot switch mode while fitting is running. Stop the fit first.",
                )
                self._act_sim.setChecked(self.state.mode == "simulation")
                self._act_fit_mode.setChecked(self.state.mode == "fitting")
                return

        self.state.mode = mode
        is_fitting = mode == "fitting"

        for fw in self._fit_widgets.values():
            fw["row_widget"].setEnabled(is_fitting)
            fw["cmb_scale"].setEnabled(is_fitting)

        self._fit_window_widget.setVisible(is_fitting)
        self.btn_fit.setEnabled(is_fitting)
        self.btn_simulate.setEnabled(not is_fitting)

        self._act_sim.setChecked(not is_fitting)
        self._act_fit_mode.setChecked(is_fitting)

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
        Req = float(self.spin_Req_um.value()) * 1e-6
        tspan = float(self.spin_tspan_us.value()) * 1e-6
        NT = int(self.spin_NT.value())
        solver = self._get_solver_settings()
        P_inf = float(self.spin_P_inf.value())
        rho = float(self.spin_rho.value())

        if key == "NHKV":
            const_kw = {k: v for k, v in const.items()
                        if k in NhkvInputs.__dataclass_fields__}
            return NhkvInputs(
                U0=params["U0"], G=params["G"], mu=params["mu"],
                Req=Req, tspan=tspan, NT=NT,
                P_inf=P_inf, rho=rho, **solver, **const_kw,
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
                damage_index=params.get("damage_index", 0),
                Req=Req, tspan=tspan, NT=NT,
                P_inf=P_inf, rho=rho, **solver, **const_kw,
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
                damage_index=params.get("damage_index", 0),
                Req=Req, tspan=tspan, NT=NT,
                P_inf=P_inf, rho=rho, **solver, **const_kw,
            )

    def _get_simulate_fn(self):
        key = self._get_active_model_key()
        if key == "NHKV":
            return simulate_nhkv_lic
        if key == "GMOD1":
            return simulate_gmod1_lic
        return simulate_gmod_lic  # GMOD2

    def _make_sim_for_fit(self, params_si: dict, tspan: float):
        """Called by the fitting engine to run one simulation."""
        key = self._get_active_model_key()
        const = dict(self._model_constants)
        Req = float(self.spin_Req_um.value()) * 1e-6
        NT = int(self.spin_NT.value())
        solver = self._get_solver_settings()
        P_inf = float(self.spin_P_inf.value())
        rho = float(self.spin_rho.value())

        if key == "NHKV":
            const_kw = {k: v for k, v in const.items()
                        if k in NhkvInputs.__dataclass_fields__}
            inp = NhkvInputs(
                U0=params_si["U0"], G=params_si["G"], mu=params_si["mu"],
                Req=Req, tspan=tspan, NT=NT,
                P_inf=P_inf, rho=rho, **solver, **const_kw,
            )
            return simulate_nhkv_lic(inp)
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
                damage_index=params_si.get("damage_index", 0),
                Req=Req, tspan=tspan, NT=NT,
                P_inf=P_inf, rho=rho, **solver, **const_kw,
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
                damage_index=params_si.get("damage_index", 0),
                Req=Req, tspan=tspan, NT=NT,
                P_inf=P_inf, rho=rho, **solver, **const_kw,
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

    def _on_fit_window_dragged(self, which: str, x_view: float):
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

        if self.state.exp_t is not None and self.state.exp_R is not None:
            t_exp = self.state.exp_t
            R_exp = self.state.exp_R
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

        if self.state.best_fit_t is not None and self.state.best_fit_R is not None:
            bf_meta = self.state.best_fit_meta
            if self.state.view_mode == "dimensional":
                t_plot = self.state.best_fit_t * 1e6
                R_plot = self.state.best_fit_R * 1e6
            else:
                if bf_meta:
                    t_plot = (
                        (self.state.best_fit_t - bf_meta.get("t_rmax", 0))
                        / bf_meta.get("tc", 1)
                    )
                    R_plot = self.state.best_fit_R / bf_meta.get("Rmax", 1)
                else:
                    t_plot = self.state.best_fit_t
                    R_plot = self.state.best_fit_R
            self.canvas.plot_fit_best(t_plot, R_plot)

        if self.state.mode == "fitting" and self.state.exp_t is not None:
            t0_view = self._time_s_to_view(
                float(self.spin_t_fit_start.value()) * 1e-6
            )
            t1_view = self._time_s_to_view(
                float(self.spin_t_fit_end.value()) * 1e-6
            )
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
    # simulation
    # =====================================================================

    def on_simulate(self):
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

        self.btn_simulate.setEnabled(False)
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
            self.btn_simulate.setEnabled(self.state.mode == "simulation")

            self.state.sim_t = out.t_sim
            self.state.sim_R = out.R_sim
            self.state.sim_meta = {"Rmax": out.Rmax_sim, "t_rmax": 0.0, "tc": out.tc}

            self._redraw_all()

            used_pinf = float(self.spin_P_inf.value())
            used_rho = float(self.spin_rho.value())
            self.lbl_output.setPlainText(
                f"Simulation ({model_key}):  Rmax={out.Rmax_sim*1e6:.3f} µm  |  "
                f"tc={out.tc*1e6:.3f} µs  |  Uc={out.Uc:.3f} m/s  |  "
                f"pts={out.t_sim.shape[0]}\n"
                f"  P_inf={used_pinf:.1f} Pa  |  rho={used_rho:.1f} kg/m³  |  "
                f"Req={float(self.spin_Req_um.value()):.3f} µm  |  NT={int(self.spin_NT.value())}"
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
            self.btn_simulate.setEnabled(self.state.mode == "simulation")
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

        sim_spec = _SimSpec(
            model_key=self._get_active_model_key(),
            Req=float(self.spin_Req_um.value()) * 1e-6,
            NT=int(self.spin_NT.value()),
            P_inf=float(self.spin_P_inf.value()),
            rho=float(self.spin_rho.value()),
            const=dict(self._model_constants),
            solver=self._get_solver_settings(),
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
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setRange(0, 0)
        dlg.show()
        self._fit_dialog = dlg

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._fit_start_time = time.time()
        self._fit_timer.start()
        self.btn_fit.setEnabled(False)

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
            elapsed = f"  |  elapsed {self._sec_to_hms(time.time() - self._fit_start_time)}"
        step_info = ""
        if prog.step_size is not None:
            step_info = f"  |  step={prog.step_size:.3e}"
        status_info = ""
        if prog.status:
            status_info = f"  [{prog.status}]"
        self.lbl_output.appendPlainText(
            f"nfev={prog.nfev}  |  LSQErr={prog.best_err:.4e}"
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
        QApplication.restoreOverrideCursor()
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

        self.btn_fit.setEnabled(self.state.mode == "fitting")

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

        self.state.best_fit_t = None
        self.state.best_fit_R = None
        self.state.best_fit_meta = None
        self._redraw_all()

        label = "Fit stopped" if was_stopped else "Fit completed"
        extra = f", elapsed {self._sec_to_hms(elapsed)}" if elapsed else ""
        self.lbl_output.appendPlainText(
            f"--- {label}{extra}  |  nfev={res.nfev}  |  LSQErr={res.lsq_err:.4e}"
        )
        self.statusBar().showMessage(label)

    def _on_fit_fail(self, msg: str, tb: str):
        QApplication.restoreOverrideCursor()
        self._fit_timer.stop()
        if self._fit_start_time is not None:
            self._fit_start_time = None
        if self._fit_dialog is not None:
            self._fit_dialog.close()
            self._fit_dialog = None
        if self._fit_worker is not None:
            self._fit_worker.wait(5000)
            self._fit_worker = None
        self.btn_fit.setEnabled(self.state.mode == "fitting")
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
                    11: ["U0", "GA1", "GA2", "alpha1", "alpha2",
                         "GB1", "GB2", "beta1", "beta2", "mu", "damage_index"],
                    7:  ["U0", "GA", "alpha", "GB", "beta", "mu", "damage_index"],
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
