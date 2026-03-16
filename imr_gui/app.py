from __future__ import annotations

import traceback
from dataclasses import dataclass
import time

import numpy as np
from scipy.io import loadmat, savemat
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressDialog,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from imr_gui.imr import NhkvInputs, NhkvOutputs, simulate_nhkv_lic
from imr_gui.imr import GMODInputs, simulate_gmod_lic
from imr_gui.io import load_experiment_mat
from imr_gui.ui.mpl_canvas import MplCanvas
from imr_gui.constitutive import (
    load_nhkv_model, load_gmod_model,
    ConstitutiveParameter, ConstitutiveModel, AVAILABLE_MODELS,
)
from imr_gui.opt import FitConfig, FitProgress, FitResult, fit_nhkv_to_experiment


# ---------------------------------------------------------------------------
# spin-box helpers
# ---------------------------------------------------------------------------


class _NoWheelSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores mouse-wheel events so that scrolling
    the parent QScrollArea works instead of changing the value."""

    def wheelEvent(self, event):  # noqa: N802
        event.ignore()


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
                 parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._bounds_si = bounds_si
        self._fit_flags = fit_flags
        self._scales = scales
        self._initial_values = initial_values
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

        settings_menu = self.menuBar().addMenu("Settings")
        settings_menu.addAction("Parallelism / Optimizer (coming soon)").setEnabled(False)

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

        # ---- Physics group box ----
        physics_box = QGroupBox("Physics")
        physics_lay = QVBoxLayout(physics_box)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._cmb_model = QComboBox()
        for key in AVAILABLE_MODELS:
            self._cmb_model.addItem(key)
        self._cmb_model.setCurrentText("NHKV")
        model_row.addWidget(self._cmb_model, stretch=1)
        physics_lay.addLayout(model_row)

        row_pinf = QHBoxLayout()
        row_pinf.addWidget(QLabel("P_inf (Pa)"))
        self.spin_P_inf = _NoWheelSpinBox()
        self.spin_P_inf.setRange(1.0, 1e9)
        self.spin_P_inf.setDecimals(1)
        self.spin_P_inf.setValue(101325.0)
        self.spin_P_inf.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        row_pinf.addWidget(self.spin_P_inf, stretch=1)
        physics_lay.addLayout(row_pinf)

        row_rho = QHBoxLayout()
        row_rho.addWidget(QLabel("rho (kg/m³)"))
        self.spin_rho = _NoWheelSpinBox()
        self.spin_rho.setRange(1.0, 1e6)
        self.spin_rho.setDecimals(1)
        self.spin_rho.setValue(998.0)
        self.spin_rho.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        row_rho.addWidget(self.spin_rho, stretch=1)
        physics_lay.addLayout(row_rho)

        row_nt = QHBoxLayout()
        row_nt.addWidget(QLabel("NT (grid)"))
        self.spin_NT = _NoWheelSpinBox()
        self.spin_NT.setRange(50, 2000)
        self.spin_NT.setDecimals(0)
        self.spin_NT.setSingleStep(50)
        self.spin_NT.setValue(500)
        self.spin_NT.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        row_nt.addWidget(self.spin_NT, stretch=1)
        physics_lay.addLayout(row_nt)

        # -- collapsible Advanced Solver Settings --
        self._adv_toggle = QToolButton()
        self._adv_toggle.setText("▸ Advanced Solver Settings")
        self._adv_toggle.setCheckable(True)
        self._adv_toggle.setChecked(False)
        self._adv_toggle.setStyleSheet("QToolButton { border: none; }")
        self._adv_toggle.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._adv_toggle.toggled.connect(self._toggle_advanced_solver)
        physics_lay.addWidget(self._adv_toggle)

        self._adv_solver_widget = QWidget()
        adv_lay = QVBoxLayout(self._adv_solver_widget)
        adv_lay.setContentsMargins(12, 0, 0, 0)

        row_solver = QHBoxLayout()
        row_solver.addWidget(QLabel("ODE solver"))
        self._cmb_solver = QComboBox()
        self._cmb_solver.addItems(["BDF", "Radau", "LSODA"])
        self._cmb_solver.setCurrentText("BDF")
        self._cmb_solver.setToolTip(
            "Radau: L-stable implicit RK (closest to MATLAB ode23tb)\n"
            "BDF: variable-order backward-differentiation (fast, stiff)\n"
            "LSODA: auto-switching Adams/BDF"
        )
        row_solver.addWidget(self._cmb_solver, stretch=1)
        adv_lay.addLayout(row_solver)

        row_rtol = QHBoxLayout()
        row_rtol.addWidget(QLabel("RelTol"))
        self.spin_rtol = _NoWheelSpinBox()
        self.spin_rtol.setRange(1e-14, 1e-1)
        self.spin_rtol.setDecimals(0)
        self.spin_rtol.setSpecialValueText("1e-8")
        self.spin_rtol.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._spin_rtol_line = _SciNotationSpinBox(self.spin_rtol, 1e-8)
        row_rtol.addWidget(self.spin_rtol, stretch=1)
        adv_lay.addLayout(row_rtol)

        row_atol = QHBoxLayout()
        row_atol.addWidget(QLabel("AbsTol"))
        self.spin_atol = _NoWheelSpinBox()
        self.spin_atol.setRange(1e-14, 1e-1)
        self.spin_atol.setDecimals(0)
        self.spin_atol.setSpecialValueText("1e-7")
        self.spin_atol.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._spin_atol_line = _SciNotationSpinBox(self.spin_atol, 1e-7)
        row_atol.addWidget(self.spin_atol, stretch=1)
        adv_lay.addLayout(row_atol)

        self._adv_solver_widget.setVisible(False)
        physics_lay.addWidget(self._adv_solver_widget)

        left_layout.addWidget(physics_box, stretch=0)

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

        self.canvas = MplCanvas()
        self.canvas.set_drag_callback(self._on_fit_window_dragged)
        right_layout.addWidget(self.canvas, stretch=3)

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
    # collapsible advanced solver settings
    # =====================================================================

    def _toggle_advanced_solver(self, checked: bool):
        self._adv_solver_widget.setVisible(checked)
        self._adv_toggle.setText(
            "▾ Advanced Solver Settings" if checked
            else "▸ Advanced Solver Settings"
        )

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
        # --- value row ---
        row = QHBoxLayout()
        unit_label = p.units[0].label if p.units else ""
        lbl_text = f"{p.label}" if not unit_label or unit_label == "-" else f"{p.label}"
        row.addWidget(QLabel(lbl_text))

        spin = _NoWheelSpinBox()
        spin.setRange(-1e15, 1e15)
        spin.setDecimals(6)
        spin.setValue(p.default)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        row.addWidget(spin, stretch=2)

        unit_combo: QComboBox | None = None
        unit_options = list(p.units) if p.units else []
        if len(unit_options) > 1:
            unit_combo = QComboBox()
            for u in unit_options:
                unit_combo.addItem(u.label)
            row.addWidget(unit_combo, stretch=1)
            unit_combo.currentIndexChanged.connect(
                lambda idx, name=p.name: self._on_unit_changed(name, idx)
            )
        elif unit_options:
            row.addWidget(QLabel(unit_options[0].label))

        parent_layout.addLayout(row)

        # --- fit controls row ---
        fw = QWidget()
        flay = QHBoxLayout(fw)
        flay.setContentsMargins(20, 0, 0, 0)

        chk = QCheckBox("Fit")
        chk.setChecked(p.fit_default)
        flay.addWidget(chk)

        flay.addWidget(QLabel("lb"))
        spin_lb = _NoWheelSpinBox()
        spin_lb.setRange(-1e15, 1e15)
        spin_lb.setDecimals(6)
        spin_lb.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin_lb.setValue(p.lb)
        flay.addWidget(spin_lb, stretch=1)

        flay.addWidget(QLabel("ub"))
        spin_ub = _NoWheelSpinBox()
        spin_ub.setRange(-1e15, 1e15)
        spin_ub.setDecimals(6)
        spin_ub.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin_ub.setValue(p.ub)
        flay.addWidget(spin_ub, stretch=1)

        flay.addWidget(QLabel("scale"))
        cmb_scale = QComboBox()
        cmb_scale.addItems(["lin", "log"])
        if p.scale == "log":
            cmb_scale.setCurrentIndex(1)
        flay.addWidget(cmb_scale)

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
        return dict(
            solver_method=self._cmb_solver.currentText(),
            rel_tol=self._spin_rtol_line.value(),
            abs_tol=self._spin_atol_line.value(),
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
        else:  # GMOD
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
        return simulate_nhkv_lic if key == "NHKV" else simulate_gmod_lic

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
        else:  # GMOD
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
        if self.state.R_eq is None or self.state.P_inf is None or self.state.rho is None:
            QMessageBox.warning(
                self, "Missing physics",
                "Please load data containing R_eq, Pinf and rho.",
            )
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

        cfg = FitConfig(
            t_exp=t_windowed,
            R_exp=R_windowed,
            make_sim=self._make_sim_for_fit,
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
            cfg, bounds_si, fit_flags, scales, initial_values, self,
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
            m = loadmat(path, squeeze_me=True, struct_as_record=False)
            if "struct_best_fit" not in m:
                raise ValueError("MAT file does not contain 'struct_best_fit'")
            sb = m["struct_best_fit"]
            flat = np.ravel(sb)

            def _to_str(rec, field):
                val = getattr(rec, field)
                arr = np.asarray(val)
                return str(arr.item()).strip() if arr.ndim == 0 else str(arr.squeeze()).strip()

            def _to_float(rec, field):
                val = getattr(rec, field)
                arr = np.asarray(val).astype(float).ravel()
                if arr.size == 0:
                    raise ValueError(f"Field '{field}' is empty")
                return float(arr[0])

            known_names = set(self._param_rows.keys())
            values: dict[str, float] = {}
            param_bounds: dict[str, dict] = {}

            for rec in flat:
                name = _to_str(rec, "name")
                if name not in known_names:
                    continue
                value = _to_float(rec, "value")
                lb = _to_float(rec, "lb")
                ub = _to_float(rec, "ub")
                scale = _to_str(rec, "scale")
                values[name] = value
                param_bounds[name] = {"lb": lb, "ub": ub, "scale": scale}

            # fallback: positional mapping
            if not values and flat.size >= len(known_names):
                ordered = list(self._param_rows.keys())
                for idx, target in enumerate(ordered):
                    if idx >= flat.size:
                        break
                    rec = flat[idx]
                    values[target] = _to_float(rec, "value")
                    param_bounds[target] = {
                        "lb": _to_float(rec, "lb"),
                        "ub": _to_float(rec, "ub"),
                        "scale": _to_str(rec, "scale"),
                    }

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
                raise ValueError("Could not map struct_best_fit to current model parameters.")
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
