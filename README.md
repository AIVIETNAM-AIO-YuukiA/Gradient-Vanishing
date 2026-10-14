# Understanding Gradient Vanishing in Recurrent Networks

This project studies a first-element memorization task: an RNN, LSTM, or
handwritten GRU receives a sequence with one sampled value at the first step
and zeros thereafter, then predicts that value at the final step. It preserves
the experiments in [GradientVanishingAnalysis.ipynb](Notebooks/GradientVanishingAnalysis.ipynb),
including their initialization, random-seed order, training behavior, and
hidden-state gradient definition. The notebook is the unchanged reference.

The package separates synthetic data (`data.py`), recurrent equations
(`models.py`), seeded training (`training.py`), gradient diagnostics
(`gradients.py`), experiments (`experiments.py`), saved-result reporting
(`reporting.py`), and the CLI/run manifest (`cli.py`, `run_manifest.py`).
The baseline fixture and tests live under `tests/`; research and usage notes
live under `docs/`.

## Install and run

The verified reference is Windows, Python 3.13.0, PyTorch 2.11.0+cpu, and the
versions in [requirements-cpu.lock](requirements-cpu.lock). Install in a new
virtual environment from a locally built wheel using the exact commands in
[installation.md](docs/installation.md). Check the CLI with
`gradient-vanishing --help` or `python -m gradient_vanishing --help`.

The checked-in [reduced CPU configuration](configs/ticket13_reduced_cpu.json)
is an explicit, short validation run. From the checkout after installation:

```powershell
gradient-vanishing main --config configs/ticket13_reduced_cpu.json --output-dir runs
gradient-vanishing gradients --config configs/ticket13_reduced_cpu.json --output-dir runs
gradient-vanishing ablation --config configs/ticket13_reduced_cpu.json --output-dir runs
gradient-vanishing sweep --config configs/ticket13_reduced_cpu.json --output-dir runs
```

`main` includes Optuna tuning, then trains default and tuned models. Each
command creates a unique run directory with real JSON results and a manifest.
Use the paths in those manifests with `gradient-vanishing report --source
KIND=PATH` to regenerate tables and PNG/SVG figures without running models.
[commands.md](docs/commands.md) gives the exact source syntax, override
precedence, result layout, and deliberate `--allow-full` safeguard.

The package defaults match the notebook's full budgets; running them is an
intentional, expensive choice. **A completed reduced run validates the
software workflow, not the notebook's full empirical results.** The full
matrix and GPU behavior are NOT VERIFIED; [Ticket 13 validation](docs/end_to_end_validation.md)
records exactly what was checked. The known GRU terminology, evaluation-mode
side effect, and missing learning-curve uncertainty band are documented in the
[approved specification](docs/refactoring_spec.md). They remain separate from
this extraction.

## Interpret the outputs

The diagnostic curves measure the batch mean L2 norm of the gradient of the
final prediction MSE with respect to each **hidden state**. They are not
parameter-gradient norms. [Gradient diagnostics](docs/gradient_diagnostics.md)
defines timestep order and the decay fit. The numeric retention-bias sweep
measures **initial** gradient decay, without per-bias trained validation or
test MSE. [Reporting](docs/reporting.md) defines each table and figure and
explains missing-condition labels. Keep `manifest.json`, raw result JSON, and
`report_index.json` with any figure used in a research report so its source
and reduced/full status remain traceable.
