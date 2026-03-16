## IMR GUI (Python prototype)

This is a Python prototype to replace the current MATLAB `patternsearch + IMR` workflow.

### Current status

- Load experimental data from a `.mat` file (expects 1D arrays `t` and `R`, with keyword-based fallback).
- Run a single **NHKV** IMR simulation (ported from `fun_IMR_NHKV.m`) and preview `R(t)` against experimental data.

### Run

1. Create a Python environment (3.10+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch:

```bash
python -m imr_gui
```

### Notes

- Units: experimental data is assumed **seconds** and **meters** internally; the GUI can display in µs/µm.
- The NHKV solver is stiff; the prototype uses SciPy `solve_ivp(..., method="BDF")`.

