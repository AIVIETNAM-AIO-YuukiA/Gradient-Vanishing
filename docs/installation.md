# Install the pinned CPU environment

The reference environment was verified on Windows 11 with Python 3.13.0 and
CPU PyTorch 2.11.0+cpu. From the project checkout, these commands create a
fresh environment, install the exact [CPU lock file](../requirements-cpu.lock),
build a non-editable wheel, and install it:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-cpu.lock
.venv\Scripts\python.exe -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
.venv\Scripts\python.exe -m pip install --no-deps dist/gradient_vanishing_research-0.1.0-py3-none-any.whl
.venv\Scripts\python.exe -m pip check
.venv\Scripts\gradient-vanishing.exe --help
.venv\Scripts\python.exe -m gradient_vanishing --help
```

The pinned lock records 38 exact dependency versions, including NumPy 2.4.0,
Optuna 4.8.0, and Matplotlib 3.10.8. A wheel-installed import and both CLI
help commands were verified from outside the checkout. The default JSON is
bundled in the wheel; verify it with:

```powershell
.venv\Scripts\python.exe -m gradient_vanishing.config
```

The package defaults reproduce the notebook's audited settings, including
three sequence lengths, three models, five comparison seeds, 20 Optuna trials
per study, 300 final training updates, and 150 tuning updates. This command
prints the effective defaults without starting an experiment. An explicit
partial configuration file and repeatable `--set KEY=JSON` options override
only named values; unknown keys and invalid values fail validation.

See [commands.md](commands.md) for the supported stages, a reduced-run example,
output paths, and the full-run safeguard. See [Ticket 13 validation](end_to_end_validation.md)
for the isolated wheel-install evidence and reproduced commands. The pinned
environment was CPU only; no GPU execution was verified. The original
[notebook](../Notebooks/GradientVanishingAnalysis.ipynb) and immutable
[baseline fixture](../tests/fixtures/notebook_baseline.json) remain unchanged.
