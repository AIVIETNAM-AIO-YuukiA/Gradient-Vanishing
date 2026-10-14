# Understanding Gradient Vanishing in Recurrent Networks

This repository provides a modular, fully tested research package reproducing the experiments in [`Notebooks/GradientVanishingAnalysis.ipynb`](Notebooks/GradientVanishingAnalysis.ipynb).

The project studies a **first-element memorization task**: a recurrent neural network (Vanilla RNN, LSTM, or handwritten GRU) processes a sequence with a single scalar value introduced at $t=0$ followed by zeros, and must predict that initial value at the final timestep. It investigates how recurrent equations, BPTT length, and gate bias initialization influence long-range gradient flow and training dynamics.

---

## Documentation

The project documentation is organized into three comprehensive documents:

* **[User & Operations Guide](docs/usage.md):** Complete installation commands, CLI reference (`main`, `gradients`, `ablation`, `sweep`, `report`), configuration override semantics, run manifests, and report generation workflows.
* **[Scientific Methodology](docs/methodology.md):** Formal problem definition, mathematical recurrence equations, BPTT hidden-gradient tracking, exponential decay curve fitting, Optuna hyperparameter optimization, and ablation designs.
* **[Full Reproducibility Report](docs/full_reproducibility_report.md):** Empirical evidence from the complete multi-hour CPU experiment matrix (180 Optuna trials, 90 main runs, 35 ablation runs, 270 gradient evaluations), artifact inventories, and comparison with historical notebook outputs.

---

## Quick Start

### 1. Installation

The canonical environment uses **Python 3.13.0**, **PyTorch 2.11.0 (CPU)**, and **Optuna 5.0.0** pinned in [`requirements-cpu.lock`](requirements-cpu.lock):

```powershell
# Create environment and install dependencies
py -3.13 -m venv .venv
& .venv\Scripts\Activate.ps1
python -m pip install -r requirements-cpu.lock

# Build and install wheel package
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
python -m pip install --no-deps (Get-Item dist\gradient_vanishing_research-*.whl).FullName
```

### 2. Run a Reduced Validation Pipeline

Execute short end-to-end experiment runs on CPU using the committed reduced configuration:

```powershell
gradient-vanishing main --config configs/ticket13_reduced_cpu.json --output-dir runs
gradient-vanishing gradients --config configs/ticket13_reduced_cpu.json --output-dir runs
gradient-vanishing ablation --config configs/ticket13_reduced_cpu.json --output-dir runs
gradient-vanishing sweep --config configs/ticket13_reduced_cpu.json --output-dir runs
```

### 3. Generate Reports from Saved Results

Generate summary tables and publication-quality plots from saved run outputs without retraining:

```powershell
gradient-vanishing report --output-dir runs `
  --source main=runs/<main-run-dir>/results/main.json
```

See [`docs/usage.md`](docs/usage.md) for full syntax and source options.

---

## Repository Structure

```text
Gradient-Vanishing/
├── configs/                            # Example and reduced configuration files
├── docs/
│   ├── full_reproducibility_report.md  # Definitive empirical reproduction report
│   ├── methodology.md                  # Scientific formulations and experimental design
│   └── usage.md                        # Installation, CLI, and reporting operations
├── Notebooks/
│   └── GradientVanishingAnalysis.ipynb # Immutable reference notebook (SHA-256: 71ec17cd...)
├── src/gradient_vanishing/             # Modular package source code
│   ├── cli.py                          # Unified CLI entry point
│   ├── config.py                       # Configuration loader and runtime validator
│   ├── data.py                         # First-element memorization data generator
│   ├── experiments.py                  # Experiment orchestration (tuning, main, ablation, sweep)
│   ├── gradients.py                    # Hidden-state gradient tracking and decay fitting
│   ├── models.py                       # Recurrent model definitions (RNN, LSTM, GRU)
│   ├── reporting.py                    # Headless table and figure rendering
│   ├── run_manifest.py                 # Provenance tracking and manifest generation
│   └── training.py                     # Deterministic training loop and evaluation
├── tests/                              # Automated test suite (136 regression tests)
│   ├── fixtures/
│   │   └── notebook_baseline.json      # Canonical execution oracle fixture
│   └── test_*.py                       # Component and integration tests
├── pyproject.toml                      # Package build configuration (Optuna 5.0.0)
├── requirements-cpu.lock               # Canonical pinned CPU dependency lockfile
├── setup.py                            # Custom build hook for wheel revision provenance
├── AGENTS.md                           # AI assistant guidelines and architectural constraints
└── README.md                           # Project overview and quick-start guide
```

---

## Development History

The initial 14-ticket refactoring, local issue tracker (`.scratch/`), and intermediate planning specifications are durably preserved in the repository Git history:
* **Archival Branch:** `archive/refactoring-docs-and-tickets`
* **Archival Tag:** `archive-refactor-history`
