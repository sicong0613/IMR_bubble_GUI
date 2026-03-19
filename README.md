# IMR Bubble GUI

A Python desktop application for inertial microrheology (IMR) bubble simulation and parameter fitting, replacing the original MATLAB `patternsearch + IMR` workflow.

---

## Features

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

- **Auto ODE tolerance** — switching to GMOD1/GMOD2 automatically sets `rtol = atol = 1e-9` (required for accurate resolution of the stiff Maxwell branch); switching to NHKV uses `1e-8 / 1e-7`.

---

## Requirements

- Python 3.10 or newer (tested on 3.11)
- See [requirements.txt](requirements.txt)

---

## Installation

```bash
pip install -r requirements.txt
```

For CMA-ES support (optional):

```bash
pip install cma
```

---

## Running

```bash
python -m imr_gui
```

On Windows with multiple Python versions installed:

```bash
py -3.11 -m imr_gui
```

---

## Loading experimental data

File → Load experiment data (.mat)

Expects a `.mat` file with 1-D arrays named `t` (seconds) and `R` (meters). Falls back to positional detection if exact names are not found. Time is automatically shifted so that the interpolated R-peak sits at t = 0.

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
