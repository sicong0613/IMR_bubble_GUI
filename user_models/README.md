# User Model Plugins

Place custom constitutive models in this directory as a JSON descriptor plus a
Python solver module. The GUI scans `user_models/*.json` at startup and adds
valid entries to the Model dropdown.

Minimal JSON shape:

```json
{
  "id": "MY_MODEL",
  "display_name": "My custom model",
  "description": "Short description.",
  "solver_entrypoint": "my_model:simulate",
  "parameters": [
    {
      "name": "G",
      "label": "G",
      "units": [{"label": "Pa", "factor": 1.0}],
      "default": 10000.0,
      "lb": 100.0,
      "ub": 1000000.0,
      "scale": "log",
      "fit_default": true
    }
  ],
  "constants": {}
}
```

## Python Solver Module

The Python file named in `solver_entrypoint` contains the actual simulation
code. The entrypoint format is:

```text
"solver_entrypoint": "module_name:function_name"
```

For example, `"my_model:simulate"` means:

- load `user_models/my_model.py`
- call the function named `simulate`

The function must have this exact signature:

```python
def simulate(params_si: dict, tspan: float, context: dict):
    ...
```

### Inputs

`params_si`

A dictionary containing the current model parameters in SI units. The keys are
the parameter names from the JSON file.

Example:

```python
{
    "G": 10000.0,      # Pa
    "mu": 0.226,       # Pa*s
    "alpha": 1.0
}
```

`tspan`

The requested physical simulation duration in seconds. During fitting, the GUI
chooses this from the fitting-window duration. During direct simulation, this is
the GUI `tspan` value.

`context`

A dictionary containing GUI/global settings that are not fitting parameters:

- `Req` in meters
- `NT`, the requested radial/grid resolution
- `P_inf` in Pa
- `rho` in kg/m^3
- `gamma` surface tension in N/m
- `c_long` longitudinal sound speed in m/s
- `bubble_model`, either `"Keller-Miksis"` or `"Rayleigh-Plesset"`
- `tspan`, the requested simulation duration in seconds. This is also passed
  as the second function argument.
- `constants`, copied from the JSON `constants` block. This is where
  model-specific physical constants such as ambient temperature (`T_inf`),
  vapor/air constants, and damage constants should be placed.
- `solver`, containing `solver_method`, `rel_tol`, and `abs_tol`
- `Rmax_exp`, the experimental peak radius in meters when available
- `model_key`, the model id from the JSON

Example:

```python
Req = float(context["Req"])
NT = int(context["NT"])
P_inf = float(context["P_inf"])
rho = float(context["rho"])
constants = dict(context.get("constants", {}))
gamma = float(context.get("gamma", 0.056))        # N/m
c_long = float(context.get("c_long", 1485.0))     # m/s
T_inf = float(constants.get("T_inf", 298.15))     # K
solver_settings = dict(context.get("solver", {}))
```

### Physical Parameters

There are two places where physical parameters can enter a plugin solver:

1. Top-level `context` fields supplied by the GUI:
   - `Req`
   - `P_inf`
   - `rho`
   - `gamma`
   - `c_long`
   - `NT`
   - `bubble_model`

2. JSON `constants`, available as `context["constants"]`:
   - `T_inf` ambient temperature in K
   - thermodynamic constants such as `D0`, `kappa`, `Ru`, `P_ref`, `T_ref`
   - model-specific constants such as `MT`, `xi_constant`, or damage thresholds

At the moment, common experiment-level physical parameters such as `P_inf`,
`rho`, `gamma`, and `c_long` are supplied by the GUI. Model-specific constants
still belong in the JSON `constants` block:

```json
"constants": {
  "T_inf": 298.15
}
```

If a physical quantity should be fit or edited per experiment, put it in the
JSON `parameters` list instead of `constants`. If it should stay fixed for that
model, keep it in `constants`.

### Bubble Dynamics Switch

The GUI exposes the Physics menu bubble-dynamics selection through:

```python
bubble_model = context.get("bubble_model", "Keller-Miksis")
```

The value is one of:

- `"Keller-Miksis"`
- `"Rayleigh-Plesset"`

Plugin solvers must decide how to use this value. The GUI does not rewrite a
plugin's RHS automatically. A plugin may:

- implement both equations and switch internally,
- support only one equation and raise an error for the other, or
- deliberately ignore the switch if the model is not bubble-dynamics-specific.

Example pattern:

```python
if context.get("bubble_model") == "Rayleigh-Plesset":
    # use RP acceleration equation
    ...
else:
    # use Keller-Miksis acceleration equation
    ...
```

### Current Shared Interface Checklist

These are the common GUI-controlled values currently passed to plugin solvers:

| Field | Location | Meaning |
|-------|----------|---------|
| `Req` | `context["Req"]` | equilibrium/reference radius, m |
| `tspan` | function argument and `context["tspan"]` | requested simulation duration, s |
| `P_inf` | `context["P_inf"]` | ambient pressure, Pa |
| `rho` | `context["rho"]` | liquid density, kg/m^3 |
| `gamma` | `context["gamma"]` | surface tension, N/m |
| `c_long` | `context["c_long"]` | longitudinal sound speed, m/s |
| `NT` | `context["NT"]` | requested grid/resolution setting |
| `bubble_model` | `context["bubble_model"]` | `"Keller-Miksis"` or `"Rayleigh-Plesset"` |
| `Rmax_exp` | `context["Rmax_exp"]` | experimental peak radius if available, m |
| `solver_method` | `context["solver"]["solver_method"]` | ODE method name |
| `rel_tol` | `context["solver"]["rel_tol"]` | ODE relative tolerance |
| `abs_tol` | `context["solver"]["abs_tol"]` | ODE absolute tolerance |

Model-specific constants such as `T_inf`, vapor constants, damage settings, or
extra grid sizes still belong in JSON `constants` unless they need a shared GUI
control.

### Output

The solver must return `imr_gui.imr.NhkvOutputs`.

Despite the name, `NhkvOutputs` is **not NHKV-specific**. It is the historical
name of the shared output container used by the GUI. Plugin solvers for any
constitutive model should return it.

Required fields:

- `t_sim`: 1-D NumPy array of simulation time in seconds, usually shifted so
  that the simulated peak radius occurs at `t = 0`
- `R_sim`: 1-D NumPy array of bubble radius in meters
- `U_sim`: 1-D NumPy array of bubble wall velocity in m/s
- `P_sim`: 1-D NumPy array of bubble pressure in Pa
- `t_sim_nondim`: 1-D NumPy array of nondimensional time
- `R_sim_nondim`: 1-D NumPy array of nondimensional radius
- `Rmax_sim`: scalar maximum simulated radius in meters
- `tc`: scalar characteristic time in seconds
- `Uc`: scalar characteristic velocity in m/s
- `n_damaged`: optional integer, use `0` if not applicable

The arrays should all have the same length and contain finite numeric values.
Fitting uses `t_sim` and `R_sim` for interpolation against experimental data.
Plotting and export use the other fields.

### Minimal Solver Example

This example is intentionally simple. It does not implement real bubble
physics; it only shows the required interface and return shape.

```python
import numpy as np

from imr_gui.imr import NhkvOutputs


def simulate(params_si: dict, tspan: float, context: dict):
    Req = float(context["Req"])
    P_inf = float(context["P_inf"])
    rho = float(context["rho"])
    NT = max(int(context.get("NT", 200)), 20)

    # Read parameters from JSON-defined names.
    amp = float(params_si.get("amp", 0.2))
    freq = float(params_si.get("freq", 1.0e5))

    # Build a toy R(t) curve.
    t = np.linspace(0.0, float(tspan), NT)
    R = Req * (1.0 + amp * np.exp(-freq * t) * np.cos(2.0 * np.pi * freq * t))
    U = np.gradient(R, t)
    P = np.full_like(t, P_inf)

    # Shift time so that the simulated R peak is at t = 0.
    i_peak = int(np.argmax(R))
    t_shifted = t - t[i_peak]

    Rmax = float(np.max(R))
    Uc = float(np.sqrt(P_inf / rho))
    tc = Req / Uc if Uc > 0 else 1.0

    return NhkvOutputs(
        t_sim=t_shifted.astype(float),
        R_sim=R.astype(float),
        U_sim=U.astype(float),
        P_sim=P.astype(float),
        t_sim_nondim=(t_shifted * Uc / Rmax).astype(float),
        R_sim_nondim=(R / Rmax).astype(float),
        Rmax_sim=Rmax,
        tc=float(tc),
        Uc=float(Uc),
        n_damaged=0,
    )
```

### Practical Notes

- Keep all internal units SI: seconds, meters, Pascals, kg/m^3.
- Raise an exception if the solver cannot produce a valid curve. During
  fitting, failed simulations are treated as large-error evaluations.
- For now, plugin models are easiest to use with `n_workers = 1`. The GUI can
  call plugin solvers during fitting, but multiprocessing support depends on
  whether the plugin module can be imported cleanly in worker processes.
- The plugin solver is responsible for implementing the chosen bubble dynamics
  if it wants to distinguish Keller-Miksis and Rayleigh-Plesset. The selected
  value is provided in `context["bubble_model"]`.

## Optional Shared ODE Helper

Plugin solvers may call SciPy directly, use another package, or use compiled
code. For simple `solve_ivp`-based models, the GUI also provides a small helper:

```python
from imr_gui.plugin_helpers import shared_ode_solver, make_outputs
```

`shared_ode_solver()` reads `solver_method`, `rel_tol`, and `abs_tol` from
`context["solver"]`, passes optional `jac_sparsity` only when appropriate for
BDF/Radau, checks `sol.success`, and returns the SciPy `OdeResult`.

Example:

```python
import numpy as np

from imr_gui.plugin_helpers import shared_ode_solver, make_outputs


def simulate(params_si: dict, tspan: float, context: dict):
    Req = float(context["Req"])
    P_inf = float(context["P_inf"])
    rho = float(context["rho"])
    Uc = float(np.sqrt(P_inf / rho))
    tc = Req / Uc

    def rhs(t, y):
        R, U = y
        omega = float(params_si.get("omega", 1.0e5))
        return [U, -omega * omega * (R - Req)]

    sol = shared_ode_solver(
        rhs=rhs,
        y0=[Req * 1.2, 0.0],
        t_span=(0.0, float(tspan)),
        context=context,
    )

    t = sol.t
    R = sol.y[0]
    U = sol.y[1]
    P = np.full_like(t, P_inf)
    return make_outputs(t_sim=t, R_sim=R, U_sim=U, P_sim=P, tc=tc, Uc=Uc)
```

This helper is for convenience and consistency, not a guaranteed speedup. It
does not replace custom pre-scans, event logic, adaptive grids, damage-history
handling, Numba, Cython, or C++ backends. If a model needs special numerical
control, write that logic inside the plugin solver.

## Plugin Contract and Future Backends

The GUI treats each model solver as a black box behind a stable interface:

```python
simulate(params_si: dict, tspan: float, context: dict) -> NhkvOutputs
```

Keep this contract stable even if the solver is later accelerated with Numba,
Cython, C++, Fortran, or another backend. The GUI should not need to know which
language implements the actual equations.

Hard rules for plugin compatibility:

1. Inputs must always come from `params_si`, `tspan`, and `context`.
2. Outputs must always be wrapped in `NhkvOutputs`.
3. Internal units must always be SI.
4. `t_sim` should be in seconds and shifted so the simulated peak radius is at
   `t = 0`.
5. `R_sim` must be in meters.
6. All output arrays should be one-dimensional, finite, and the same length.
7. The solver must not read GUI widgets or depend on `MainWindow`.
8. The solver should be deterministic for the same inputs.
9. Invalid parameter sets should raise an exception or return a clearly invalid
   result, not silently produce misleading curves.
10. Fitting parameters should not be hard-coded inside the solver; they should
    be read from `params_si`.

Recommended model layout for future maintainability:

```text
user_models/
  my_model.json
  my_model.py
```

For accelerated backends, keep the same Python entrypoint as a thin wrapper:

```text
user_models/
  my_model.json
  my_model.py          # defines simulate(), calls backend if available
  my_model_backend.pyd # optional compiled extension on Windows
```

Example wrapper pattern:

```python
from imr_gui.imr import NhkvOutputs
from my_model_backend import run_solver


def simulate(params_si: dict, tspan: float, context: dict):
    raw = run_solver(params_si, tspan, context)
    return NhkvOutputs(
        t_sim=raw.t_sim,
        R_sim=raw.R_sim,
        U_sim=raw.U_sim,
        P_sim=raw.P_sim,
        t_sim_nondim=raw.t_sim_nondim,
        R_sim_nondim=raw.R_sim_nondim,
        Rmax_sim=raw.Rmax_sim,
        tc=raw.tc,
        Uc=raw.Uc,
        n_damaged=getattr(raw, "n_damaged", 0),
    )
```

This allows individual models to move from Python to compiled code one at a
time without changing the GUI, fitting engine, job list, or export logic.
