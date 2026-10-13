# Install and check configuration

This package is being extracted from the unchanged [source notebook](../Notebooks/GradientVanishingAnalysis.ipynb). Ticket 01 provides only project installation and validated experiment configuration. The notebook remains the algorithmic baseline; saved notebook outputs are not independently verified results.

## Reference CPU environment

Use Python 3.13 on a CPU machine. The reference environment was resolved on Windows with Python 3.13.0. Install the exact versions in `requirements-cpu.lock`, then the package:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-cpu.lock
.venv\Scripts\python -m pip install --no-deps .
.venv\Scripts\python -m gradient_vanishing.config
```

The lock file records the installed direct dependencies and their active transitive dependencies for this reference environment. Python and platform matter for Torch wheels; use a CPU Torch build matching the pinned version if the platform's default wheel includes CUDA support. CUDA is not part of this reference environment. Future CUDA runs must record the CUDA runtime and hardware in run metadata.

The package default is `src/gradient_vanishing/configs/default.json`. It is bundled into installed wheels, so the config check works outside the checkout. A partial JSON file overrides only its named keys:

```powershell
python -m gradient_vanishing.config path/to/partial.json
```

Code that invokes a run can call `load_config(path, overrides)`; precedence is command overrides, then file values, then package defaults. Unknown keys and invalid values raise `ConfigError`. The JSON schema version is `1`. The notebook's `EPOCHS` means optimizer updates here; `gradient_capture_updates` remains zero-based to match the notebook loop. No experiment entry point is implemented in Ticket 01.

Future experiment entry points should call `prepare_runtime(config)` before importing Torch. For `auto` or `cuda`, it sets the notebook's `CUBLAS_WORKSPACE_CONFIG=:4096:8`; calling it after Torch import raises `ConfigError`. Torch deterministic algorithms and seeding belong to the later runtime ticket.

Run the focused checks with `python -m pytest tests/test_config.py -q`.
