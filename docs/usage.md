# User and Operations Guide

This guide describes how to install the verified environment, execute research experiment stages via the `gradient-vanishing` CLI, configure execution budgets, inspect run provenance manifests, and generate publication-ready tables and figures from saved results.

---

## 1. Environment and Installation

The canonical execution environment uses **Python 3.13.0**, **PyTorch 2.11.0 (CPU)**, and **Optuna 5.0.0**. The exact dependencies are pinned in [`requirements-cpu.lock`](../requirements-cpu.lock).

### 1.1. Fresh Virtual Environment Setup

From the repository checkout, execute:

```powershell
# Create and activate a dedicated virtual environment
py -3.13 -m venv .venv
& .venv\Scripts\Activate.ps1

# Install pinned dependencies
python -m pip install -r requirements-cpu.lock

# Build a non-editable wheel and install it
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
python -m pip install --no-deps (Get-Item dist\gradient_vanishing_research-*.whl).FullName

# Verify environment and CLI entry points
python -m pip check
gradient-vanishing --help
python -m gradient_vanishing --help
```

### 1.2. Source Revision and Build Provenance

Wheels built via `pip wheel` automatically invoke `BuildWithRevision` in [`setup.py`](../setup.py), embedding the current Git commit hash into `gradient_vanishing/_build_info.py`. When experiment runs are launched outside the repository checkout, the CLI records this embedded revision in `manifest.json` under `git_commit`.

### 1.3. Default Configuration Inspection

The package bundles audited default experiment parameters matching the original notebook:

```powershell
python -m gradient_vanishing.config
```

This outputs the effective configuration dictionary without initiating training.

---

## 2. Command-Line Interface

The project exposes a unified command-line tool `gradient-vanishing` (or `python -m gradient_vanishing`):

```text
usage: gradient-vanishing [-h] {main,gradients,ablation,sweep,report} ...
```

### Available Stages

| Stage | Purpose | Primary Output Artifact |
|---|---|---|
| `main` | Runs Optuna tuning for each architecture, then trains default and tuned models across comparison seeds | `results/main.json`, `results/study_*.json` |
| `gradients` | Evaluates initial hidden-state gradient norms across timesteps using full BPTT in float64 | `results/initial_gradient_*.json` |
| `ablation` | Evaluates 7 gate-initialization modes at $T=100$ and tracks time-to-threshold | `results/gate_ablation.json` |
| `sweep` | Sweeps numeric retention biases ($b \in [-4.0, +4.0]$) and measures initial gradient decay | `results/retention_bias_sweep.json` |
| `report` | Generates summary CSV tables and SVG/PNG figures strictly from saved JSON records | `reports/report_index.json`, tables, figures |

---

## 3. Configuration and Override Precedence

Every CLI command accepts configuration overrides:

```powershell
gradient-vanishing <stage> [--config PATH.json] [--set KEY=JSON] [--output-dir DIR] [--allow-full]
```

### Precedence Hierarchy

1. **Checked-in Defaults:** [`src/gradient_vanishing/configs/default.json`](../src/gradient_vanishing/configs/default.json).
2. **Configuration File:** Specified via `--config PATH.json` (partial JSON files merge recursively).
3. **Command-Line Overrides:** Specified via repeatable `--set KEY=JSON` (e.g., `--set training.updates=10`).
4. **Output Directory:** Specified via `--output-dir DIR`.

### Safety Guardrail: `--allow-full`

The package defaults describe the full notebook-size matrix (3 sequence lengths, 3 models, 5 seeds, 20 Optuna trials per model/length, 300 updates). Because this full execution takes multiple hours on CPU, running default full configurations requires an explicit `--allow-full` flag. Without this flag, commands safely abort before creating directories or allocating memory.

### Running a Reduced Validation Pipeline

For fast verification on CPU, use the committed configuration [`configs/ticket13_reduced_cpu.json`](../configs/ticket13_reduced_cpu.json):

```powershell
gradient-vanishing main --config configs/ticket13_reduced_cpu.json --output-dir runs
gradient-vanishing gradients --config configs/ticket13_reduced_cpu.json --output-dir runs
gradient-vanishing ablation --config configs/ticket13_reduced_cpu.json --output-dir runs
gradient-vanishing sweep --config configs/ticket13_reduced_cpu.json --output-dir runs
```

---

## 4. Run Directory Layout and Manifest Contract

Each command creates an isolated, uniquely timestamped run directory under `--output-dir`:

```text
runs/
  main-20261014T162411-1791995051528387400-18468/
    manifest.json
    results/
      main.json
      study_vanilla_rnn_T10.json
      ...
  report-20261014T164059-1791996059341184600-25276/
    manifest.json
    reports/
      report_index.json
      main_summary.csv
      main_learning_curves.png (.svg)
      training_gradient_heatmap_T100.png (.svg)
```

### The `manifest.json` Provenance Contract

Every run directory contains `manifest.json` recording:
* **Identification:** Unique `run_id`, stage name, and exact CLI invocation.
* **Timestamps:** UTC start and completion timestamps.
* **Status:** `complete`, `incomplete` (with partial results preserved), or `failed`.
* **Configuration:** Merged effective configuration, explicit CLI overrides, and config-file SHA-256.
* **Runtime & Hardware:** Python version, Torch version, Optuna version, CUDA availability, deterministic algorithm status, and Torch thread count.
* **Git Commit:** Source Git SHA-256 (from repository or embedded wheel `_build_info.py`).
* **Artifact Inventory:** Relative paths, byte sizes, and SHA-256 checksums of all written results.

---

## 5. Reporting from Saved Results

The `report` stage reads existing JSON result files and renders figures and tables. It does **not** import model classes or execute training.

### Report CLI Invocation

Pass paths to saved result files using repeatable `--source KIND=PATH` flags:

```powershell
gradient-vanishing report --output-dir runs `
  --source main=runs/<main-run>/results/main.json `
  --source ablation=runs/<ablation-run>/results/gate_ablation.json `
  --source sweep=runs/<sweep-run>/results/retention_bias_sweep.json `
  --source initial_gradients=runs/<grad-run>/results/initial_gradient_vanilla_rnn_T100.json `
  --source initial_gradients=runs/<grad-run>/results/initial_gradient_lstm_T100.json `
  --source initial_gradients=runs/<grad-run>/results/initial_gradient_gru_T100.json `
  --source study=runs/<main-run>/results/study_vanilla_rnn_T10.json
```

### Generated Artifacts

| Source Kind | Generated Output | Content Description |
|---|---|---|
| `main` | `main_summary.csv` | Mean $\pm$ sample standard deviation of test MSE across comparison seeds |
| `main` | `main_learning_curves.png`, `.svg` | Semilog validation learning curves against the theoretical $1/3$ memoryless baseline |
| `main` | `training_gradient_heatmap_T{T}.png`, `.svg` | Heatmap of relative hidden-state gradient norms across training epochs |
| `ablation` | `gate_ablation.csv` | Test MSE, convergence speed (updates to threshold $< 0.01$), and initial decay factor |
| `sweep` | `retention_bias_values.csv`<br>`retention_bias_sweep_T{T}.png`, `.svg` | Measured initial gradient decay vs. theoretical $\sigma(b)$ reference |
| `study` | `optuna_trials.csv` | Trial parameters, validation objectives, and prune/complete statuses |
| All | `report_index.json` | Complete index with source hashes, scopes, and verification statuses |

Matplotlib operates using the headless `Agg` backend; no GUI or display server is required.

---

## 6. Baseline Verification and Test Execution

The repository maintains an immutable baseline fixture captured directly from original notebook code cells.

### Running Regression Tests

```powershell
# Run the complete test suite (136 tests)
pytest tests/ -q

# Run baseline oracle verification tests
pytest tests/test_notebook_baseline.py -q
```

### Recapturing Baseline Oracles

To execute the original notebook definitions and generate a fresh comparison baseline in a temporary location:

```powershell
python scripts/capture_notebook_baseline.py --output .tmp/fresh_baseline.json
```

Existing fixtures are protected against accidental overwriting by explicit guard checks in `capture_notebook_baseline.py`.
