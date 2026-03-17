from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


@dataclass
class PlotHandles:
    exp_line: object | None = None
    sim_line: object | None = None
    fit_best_line: object | None = None
    fit_window_fill: object | None = None
    fit_window_vline_start: object | None = None
    fit_window_vline_end: object | None = None


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(constrained_layout=True)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.handles = PlotHandles()

        # data bounds for zoom reference (set after each plot)
        self._data_xlim: tuple | None = None
        self._data_ylim: tuple | None = None

        self.ax.set_xlabel("t (s)")
        self.ax.set_ylabel("R (m)")
        self.ax.grid(True, alpha=0.3)

        # drag state for fit-window lines
        self._drag_target: str | None = None  # "start" or "end"
        self._drag_callback: Callable[[str, float], None] | None = None
        self._pick_tolerance = 0.02  # fraction of x-axis range
        self._drag_xlim: tuple[float, float] | None = None  # clamp range

        # pan state
        self._pan_start: tuple | None = None   # (event.x, event.y) in display coords
        self._pan_xlim: list | None = None
        self._pan_ylim: list | None = None

        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    # ---- draggable fit-window support ------------------------------------

    def set_drag_callback(self, cb: Callable[[str, float], None] | None):
        """Register *cb(which, x_data)* called when a fit-window line is
        dragged.  *which* is ``"start"`` or ``"end"``."""
        self._drag_callback = cb

    def set_drag_limits(self, x_min: float, x_max: float):
        """Clamp dragged fit-window lines to ``[x_min, x_max]``."""
        self._drag_xlim = (min(x_min, x_max), max(x_min, x_max))

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        h = self.handles

        # Check if near a fit-window line first
        if h.fit_window_vline_start is not None or h.fit_window_vline_end is not None:
            xlim = self.ax.get_xlim()
            tol = (xlim[1] - xlim[0]) * self._pick_tolerance
            x = event.xdata

            dist_start = dist_end = float("inf")
            if h.fit_window_vline_start is not None:
                dist_start = abs(x - h.fit_window_vline_start.get_xdata()[0])
            if h.fit_window_vline_end is not None:
                dist_end = abs(x - h.fit_window_vline_end.get_xdata()[0])

            best = min(dist_start, dist_end)
            if best <= tol:
                self._drag_target = "start" if dist_start <= dist_end else "end"
                return

        # Not near a fit line — start pan
        self._pan_start = (event.x, event.y)
        self._pan_xlim = list(self.ax.get_xlim())
        self._pan_ylim = list(self.ax.get_ylim())

    def _on_release(self, event):
        self._drag_target = None
        self._pan_start = None

    def _on_motion(self, event):
        # --- fit-window line drag ---
        if self._drag_target is not None:
            if event.inaxes != self.ax:
                return
            x = event.xdata
            if x is None:
                return
            if self._drag_xlim is not None:
                x = max(self._drag_xlim[0], min(x, self._drag_xlim[1]))

            vline = (
                self.handles.fit_window_vline_start
                if self._drag_target == "start"
                else self.handles.fit_window_vline_end
            )
            if vline is not None:
                vline.set_xdata([x, x])

            self._update_fill_region()
            self.draw_idle()

            if self._drag_callback is not None:
                self._drag_callback(self._drag_target, x)
            return

        # --- pan ---
        if self._pan_start is not None and event.x is not None:
            dx_disp = event.x - self._pan_start[0]
            dy_disp = event.y - self._pan_start[1]
            xlim = self._pan_xlim
            ylim = self._pan_ylim
            bbox = self.ax.get_window_extent()
            if bbox.width > 0 and bbox.height > 0:
                dx_data = -dx_disp * (xlim[1] - xlim[0]) / bbox.width
                dy_data = -dy_disp * (ylim[1] - ylim[0]) / bbox.height
                self.ax.set_xlim(xlim[0] + dx_data, xlim[1] + dx_data)
                self.ax.set_ylim(ylim[0] + dy_data, ylim[1] + dy_data)
                self.draw_idle()

    def _update_fill_region(self):
        """Redraw the shaded region between the two vlines."""
        if self.handles.fit_window_fill is not None:
            try:
                self.handles.fit_window_fill.remove()
            except Exception:
                pass
            self.handles.fit_window_fill = None

        vs = self.handles.fit_window_vline_start
        ve = self.handles.fit_window_vline_end
        if vs is not None and ve is not None:
            x0 = vs.get_xdata()[0]
            x1 = ve.get_xdata()[0]
            self.handles.fit_window_fill = self.ax.axvspan(
                min(x0, x1), max(x0, x1),
                alpha=0.08, color="gray", zorder=0,
            )

    # ---- simulation curve ------------------------------------------------

    def clear_sim(self):
        if self.handles.sim_line is not None:
            try:
                self.handles.sim_line.remove()
            except Exception:
                pass
            self.handles.sim_line = None

    def plot_experiment(self, t: np.ndarray, R: np.ndarray):
        if self.handles.exp_line is not None:
            try:
                self.handles.exp_line.remove()
            except Exception:
                pass

        (line,) = self.ax.plot(
            t,
            R,
            linestyle="None",
            marker="s",
            markersize=2,
            color="k",
            label="exp data",
        )
        self.handles.exp_line = line
        self.ax.legend(loc="best")
        self.draw_idle()

    def plot_simulation(self, t: np.ndarray, R: np.ndarray):
        self.clear_sim()
        (line,) = self.ax.plot(t, R, "-", linewidth=1.5, label="simulation", color="C0")
        self.handles.sim_line = line
        self.ax.legend(loc="best")
        self.draw_idle()

    # ---- best-fit curve (live during fitting) ----------------------------

    def plot_fit_best(self, t: np.ndarray, R: np.ndarray):
        self.clear_fit_best()
        (line,) = self.ax.plot(
            t, R, "--", linewidth=1.5, label="best fit", color="#2ca02c",
        )
        self.handles.fit_best_line = line
        self.ax.legend(loc="best")
        self.draw_idle()

    def clear_fit_best(self):
        if self.handles.fit_best_line is not None:
            try:
                self.handles.fit_best_line.remove()
            except Exception:
                pass
            self.handles.fit_best_line = None

    # ---- fit time window -------------------------------------------------

    def draw_fit_window(self, t_start: float, t_end: float):
        self.clear_fit_window()
        self.handles.fit_window_fill = self.ax.axvspan(
            t_start, t_end, alpha=0.08, color="gray", zorder=0,
        )
        self.handles.fit_window_vline_start = self.ax.axvline(
            t_start, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
        )
        self.handles.fit_window_vline_end = self.ax.axvline(
            t_end, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
        )
        self.draw_idle()

    def clear_fit_window(self):
        for attr in ("fit_window_fill", "fit_window_vline_start", "fit_window_vline_end"):
            obj = getattr(self.handles, attr, None)
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
                setattr(self.handles, attr, None)

    # ---- zoom support -------------------------------------------------------

    def set_data_bounds(self, xlim: tuple, ylim: tuple):
        """Store the full data bounds for zoom reference."""
        self._data_xlim = xlim
        self._data_ylim = ylim

    def zoom_x(self, fraction: float):
        """Zoom x-axis anchored at the left edge (x_min stays fixed)."""
        if self._data_xlim is None:
            return
        x_left = self._data_xlim[0]
        x_range = (self._data_xlim[1] - self._data_xlim[0]) * max(0.01, fraction)
        self.ax.set_xlim(x_left, x_left + x_range)
        self.draw_idle()

    def zoom_y(self, fraction: float):
        """Zoom y-axis anchored at the bottom edge (y_min stays fixed)."""
        if self._data_ylim is None:
            return
        y_bottom = self._data_ylim[0]
        y_range = (self._data_ylim[1] - self._data_ylim[0]) * max(0.01, fraction)
        self.ax.set_ylim(y_bottom, y_bottom + y_range)
        self.draw_idle()

    def reset_zoom(self):
        """Restore full data view."""
        if self._data_xlim is not None:
            self.ax.set_xlim(self._data_xlim)
        if self._data_ylim is not None:
            self.ax.set_ylim(self._data_ylim)
        self.draw_idle()

    def wheelEvent(self, event):  # noqa: N802
        """Zoom in/out centered on mouse cursor (like MATLAB)."""
        if self._data_xlim is None and self._data_ylim is None:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return

        factor = 0.85 if delta > 0 else 1.0 / 0.85
        ax = self.ax
        xlim = list(ax.get_xlim())
        ylim = list(ax.get_ylim())

        # Get mouse position in data coordinates
        try:
            pos = event.position()
            x_pix = pos.x()
            y_pix = pos.y()
            # Convert Qt top-left origin to matplotlib bottom-left origin
            y_mpl = self.height() - y_pix
            x_data, y_data = ax.transData.inverted().transform((x_pix, y_mpl))
            # Clamp to current view
            x_data = max(xlim[0], min(x_data, xlim[1]))
            y_data = max(ylim[0], min(y_data, ylim[1]))
        except Exception:
            x_data = (xlim[0] + xlim[1]) / 2
            y_data = (ylim[0] + ylim[1]) / 2

        new_xlim = [x_data + (x - x_data) * factor for x in xlim]
        new_ylim = [y_data + (y - y_data) * factor for y in ylim]
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.draw_idle()
        event.accept()
