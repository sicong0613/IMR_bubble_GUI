# IMR Fitting GUI

**Beta 1.2**

A Python desktop application for inertial microrheology (IMR) bubble simulation and parameter fitting, replacing the original MATLAB `patternsearch + IMR` workflow.

---

## Features

- **Job list workflow** - queue fitting jobs, reorder queued jobs, run the queue, stop the queue, clear completed jobs, and preview each job's experiment, fit window, and best-fit curve.

- **Batch job creation** - batch-add multiple experiment `.mat` files as fitting jobs using the current Fitting module settings as the template.

- **Job queue import / export** - save and load `.imrqueue` archives containing job metadata, experiment data, previews, and result MAT files.

- **Queue handoff option** - optionally seed each queued job from the previous job's best-fit parameters when job type and model match exactly.

- **Multiple constitutive models** — select at runtime via the Model menu:
  - **NHKV** — Neo-Hookean Kelvin-Voigt
  - **GMOD1** — Generalized Maxwell-Ogden + Damage, 1-term (single elastic + single viscous branch)
  - **GMOD2** — Generalized Maxwell-Ogden + Damage, 2-term (two elastic + two viscous branches)
  - Models are defined in JSON files under `imr_gui/constitutive/` — adding a new model requires only a JSON descriptor and a solver function.

- **Forward simulation** — run a Keller-Miksis bubble dynamics simulation for the selected model and compare against experimental `R(t)` data.

- **Parameter fitting** — least-squares curve fitting with seven optimization algorithms:
  - **Nelder-Mead** — fast local search, good for well-conditioned problems
  - **Powell** — derivative-free directional search
  - **Pattern Search** — GPS (GPSPositiveBasis2N), equivalent to MATLAB's `patternsearch`; best accuracy in practice
  - **Differential Evolution** — global stochastic search; supports true multiprocessing
  - **CMA-ES** — covariance matrix adaptation evolution strategy (requires optional `cma` package)
  - **Dual Annealing** — simulated annealing + local search
  - **Basin Hopping** — stochastic global search with local minimization
  - Per-algorithm settings (bounds, tolerances, population size, mesh parameters, etc.) in the Optimizer Settings dialog.

- **Parallel fitting** — Pattern Search and Differential Evolution support `n_workers > 1` for true multiprocessing (bypasses the Python GIL via `ProcessPoolExecutor`). Recommended for GMOD1/GMOD2 which have expensive ODE evaluations.

- **Load / Save parameters** — import fitted parameters from MATLAB `struct_best_fit` MAT files (supports both v5 char-array names and modern MATLAB `string` type via positional layout matching). Save current parameters back to `.mat`.

- **Cross-model parameter loading** — when loading a GMOD2 result into GMOD1 (or vice versa), parameter names are matched by canonical stem: `GA1 ↔ GA`, `alpha1 ↔ alpha`, `GB1 ↔ GB`, `beta1 ↔ beta`. If the primary term is zero, the secondary term is used as fallback.

- **Unit display** — spinboxes support Pa / kPa / MPa for stiffness, µs/µm for time and radius.

- **Plot controls** — zoom, time window, normalization toggle (R/Req), experimental data overlay.

- **Curve View panel** - compare many imported experiment/simulation curves at once, edit legends, line widths, visibility, curve order, and batch-apply either high-contrast distinct colors or parameter-sweep gradient colors.

- **Auto ODE tolerance** — switching to GMOD1/GMOD2 automatically sets `rtol = atol = 1e-9` (required for accurate resolution of the stiff Maxwell branch); switching to NHKV uses `1e-8 / 1e-7`.

---

## Requirements

- Python 3.11 is recommended.
- Do not install the GUI into a shared/global Python environment unless you are deliberately maintaining that environment.
- Recommended dependency entry points:
  - `requirements.txt` for a local `venv` environment.
  - `environment.yml` for Anaconda/Miniconda users.

---

## Installation

### Option A: Python venv on Windows (recommended)

If the project folder is inside OneDrive or another sync service, create the virtual environment outside the synced folder. For example:

```powershell
py -3.11 -m venv C:\venvs\imr-gui
C:\venvs\imr-gui\Scripts\activate
cd "path\to\IMR_bubble_GUI"
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the project is not in a synced folder, a project-local `.venv` also works:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then run:

```powershell
python main.py
```

or:

```powershell
python -m imr_gui
```

After the `C:\venvs\imr-gui` environment is created, Windows users can also start the GUI by double-clicking:

```text
run_imr_gui.bat
```

The batch file expects the virtual environment at `C:\venvs\imr-gui`.

### Option B: Python venv on macOS

Install Python 3.11 first, for example from python.org or Homebrew. If using Homebrew:

```bash
brew install python@3.11
```

Create the virtual environment outside synced folders such as OneDrive, iCloud Drive, or Dropbox:

```bash
python3.11 -m venv ~/venvs/imr-gui
source ~/venvs/imr-gui/bin/activate
cd /path/to/IMR_bubble_GUI
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

For later runs:

```bash
source ~/venvs/imr-gui/bin/activate
cd /path/to/IMR_bubble_GUI
python main.py
```

### Option C: Conda / Anaconda

From the repository root:

```powershell
conda env create -f environment.yml
conda activate imr-fitting-gui
python main.py
```

### Optional CMA-ES optimizer

The CMA-ES optimizer is optional. Install it only if you plan to use the CMA-ES method:

```powershell
pip install "cma>=3.3,<4"
```

For conda users, install it after activating the environment:

```powershell
conda activate imr-fitting-gui
pip install "cma>=3.3,<4"
```

---

## Running

On Windows, if you used the recommended `C:\venvs\imr-gui` environment, run:

```powershell
.\run_imr_gui.bat
```

Otherwise, activate the project environment first, then run either:

```powershell
python main.py
```

or:

```powershell
python -m imr_gui
```

The main window title should read `IMR Fitting GUI (beta 1.2)`.

---

## Tutorials

Step-by-step tutorials are available in [docs/README.md](docs/README.md).

Current tutorial topics include:

- [Import experiment data](docs/tutorials/import-data.md)

---

## Environment Notes

- `.venv/`, `venv/`, and `env/` are ignored by git.
- Git ignore rules do not stop OneDrive from syncing virtual environments. If this repository is in OneDrive, keep the venv in a local folder such as `C:\venvs\imr-gui`.
- On Windows, `run_imr_gui.bat` is a convenience launcher for the recommended `C:\venvs\imr-gui` environment.
- `requirements.txt` intentionally pins dependency ranges rather than using unbounded latest versions; GUI libraries such as PySide6 and Matplotlib can change behavior across major/minor releases.
- `environment.yml` is provided for users who prefer Anaconda/Miniconda, but the lightweight `venv` workflow is the primary development path.
- Future PyInstaller builds should be created from a clean virtual environment, not from a global Python installation.

---

## Loading experimental data

File → Load experiment data (.mat)

Expects a `.mat` file with 1-D arrays named `t` (seconds) and `R` (meters). Falls back to positional detection if exact names are not found. Time is automatically shifted so that the interpolated R-peak sits at t = 0.

---

## Curve View and Color Palettes

Open the multi-curve comparison panel from:

```text
View -> Curve Selection Panel
```

Enable `Enable multiple curve selection`, then use `Batch import curves` to add experiment or simulation result `.mat` files. If a MAT file contains a top-level `legend` field, it is used as the curve legend. LaTeX-style Greek names such as `\alpha` are displayed as Unicode Greek letters in the GUI.

The Curve View panel supports:

- show/hide curves without deleting them
- row selection with Ctrl/Shift multi-select
- `Clear selected`, `Clear all`
- `Move to top`, `Move up`, `Move down`
- legend editing, type switching, color selection, and line-width editing

Color assignment is explicit:

1. Import curves.
2. Choose `Color mode`.
3. Click `Apply colors`.

Available color modes:

- `Distinct`: high-contrast colors for a small number of curves.
- `Sweep gradient`: parameter-sweep colors, mapped by the current curve row order.

If rows are selected, `Apply colors` only recolors selected rows. If no rows are selected, it recolors all curves. This makes it possible to reorder curves first, then apply a sweep gradient in the desired parameter direction.

Color presets live in:

```text
imr_gui/view_colors.json
```

Each entry can define:

```json
{"name": "Blue", "color": "#0072BD", "role": "sim", "palette": "distinct"}
```

Fields:

- `name`: display name / tooltip.
- `color`: hex RGB color.
- `role`: `sim`, `exp`, or `both`.
- `palette`: `distinct`, `sweep`, or `both`.

The default `view_colors.json` includes high-contrast distinct colors and a 20-color sweep gradient. If the number of curves does not match the number of sweep colors, the GUI interpolates along the sweep palette.

Curve View exports are under the `View` menu:

- `Export view (.mat)`: exports curve data and style metadata.
- `Export view (.svg)`: exports a vector figure.
- `Copy view (png)` / `Copy view (svg)`: copies the current preview to the clipboard.

---

## Loading fitted parameters

File → Load parameters (MAT)...

Expects a `.mat` file containing a `struct_best_fit` array with fields `name`, `value`, `lb`, `ub`, `scale`.

**MAT file format compatibility:**

| MATLAB save format | `name` field type | Supported |
|--------------------|-------------------|-----------|
| `-v7` / `-v5`  with `char` names | `char` array | Yes (scipy) |
| `-v7.3` (HDF5) | `string` or `char` | Yes (mat73) |
| `-v7` / `-v5` with `string` names | MCOS object | Yes (positional fallback by struct size: 11 → GMOD2 layout, 7 → GMOD1 layout) |

---

## ODE solver tolerances

GMOD1 and GMOD2 are significantly stiffer than NHKV due to the Maxwell branch (`λ_nv`) ODEs (MT = 200 material points). The BDF solver requires tight tolerances to correctly resolve these dynamics:

| Model | Recommended rtol | Recommended atol |
|-------|-----------------|-----------------|
| NHKV  | 1e-8 | 1e-7 |
| GMOD1 | 1e-9 | 1e-9 |
| GMOD2 | 1e-9 | 1e-9 |

These are set automatically when switching models. Using looser tolerances (e.g., atol = 1e-6) produces physically incorrect results (spurious underdamped oscillations). Fitting uses the same tolerances as the display simulation — always verify the tolerance settings before fitting.

---

## Project structure

```
imr_gui/
├── app.py                  # Main window, GUI logic
├── constitutive/
│   ├── nhkv.json           # NHKV model descriptor
│   ├── gmod1.json          # GMOD1 model descriptor
│   ├── gmod.json           # GMOD2 model descriptor
│   └── __init__.py         # Model registry (AVAILABLE_MODELS)
├── imr/
│   ├── nhkv.py             # NHKV solver
│   ├── gmod_solver.py      # GMOD1 / GMOD2 solvers (Keller-Miksis + BDF)
│   └── __init__.py
├── opt/
│   └── nhkv_fit.py         # Fitting engine (all algorithms)
├── io/
│   └── mat_loader.py       # Experimental data loader
└── ui/
    └── mpl_canvas.py       # Matplotlib canvas widget
Example_MATLAB/             # Reference MATLAB scripts and example data
test/                       # Test MAT files
```

---

## Notes

- All internal units are SI (seconds, meters, Pascals). Display units are cosmetic only.
- Solvers use `scipy.solve_ivp` with the BDF method and sparse Jacobian (SuperLU) for stiff bubble dynamics.
- GMOD1 and GMOD2 are independent constitutive models with separate parameter sets, solvers, and JSON descriptors.
- Parallel fitting uses `multiprocessing` (not threads) to bypass the Python GIL. On Windows, the entry point `python -m imr_gui` includes the required `freeze_support()` guard.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
