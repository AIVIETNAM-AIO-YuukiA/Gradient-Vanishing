# Project Documentation Index

This directory contains technical documentation, scientific specifications, architectural proposals, operational standards, and reproducibility reports for the **Gradient-Vanishing** research project.

The canonical execution environment uses **Python 3.13.0**, **PyTorch 2.11.0 (CPU)**, and **Optuna 5.0.0** pinned in [`requirements-cpu.lock`](../requirements-cpu.lock).

---

## 1. Guides and Usage

Operational instructions for installing the environment, building wheels, and invoking the research CLI:

* [`installation.md`](installation.md): Step-by-step instructions for creating a clean virtual environment, installing pinned dependencies via `requirements-cpu.lock`, and building/installing the non-editable wheel package.
* [`commands.md`](commands.md): Reference guide for the `gradient-vanishing` CLI tool, describing all experiment stages (`main`, `gradients`, `ablation`, `sweep`, `report`), configuration override semantics (`--set`), output directory management, and the `--allow-full` safeguard.

---

## 2. Research, Experiments, and Scientific Reports

Detailed mathematical models, experiment specifications, diagnostic definitions, and empirical validation reports:

* [`data.md`](data.md): Specification of the first-element memorization sequence task, input tensor generation, target definitions, and fixed validation/test set caching.
* [`gradient_diagnostics.md`](gradient_diagnostics.md): Mathematical definitions of intermediate hidden-state gradient tracking, full-BPTT sensitivity, and exponential decay curve fitting across recurrent steps.
* [`optuna_study.md`](optuna_study.md): Hyperparameter search space definitions, TPE sampling configuration, and median pruning rules for tuning recurrent architectures.
* [`main_comparison.md`](main_comparison.md): Design and configuration matrix for comparing Vanilla RNN, LSTM, and GRU across sequence lengths ($T \in \{10, 30, 100\}$) in default and tuned configurations.
* [`gate_ablation.md`](gate_ablation.md): Gate-initialization ablation experiment at $T=100$, analyzing the impact of `zero`, `one`, and `chrono` initializations on training speed and gradient decay.
* [`retention_bias_sweep.md`](retention_bias_sweep.md): Continuous parametric sweep of retention bias values from $-4.0$ to $+4.0$, measuring initial gradient flow and comparing with theoretical sigmoid retention.
* [`reporting.md`](reporting.md): Specifications for generating summary tables (CSV) and figures (learning curves, semilog gradient curves, heatmaps) from saved JSON results.
* [`full_experiment_execution_plan.md`](full_experiment_execution_plan.md): Preflight execution plan for running the full 14-ticket experiment matrix on CPU.
* [`full_reproducibility_report.md`](full_reproducibility_report.md): Definitive reproducibility report recording full-matrix execution results, direct notebook comparisons, artifact hashes, and historical notebook observations.

---

## 3. Development, Architecture, and Refactoring Specifications

Architecture proposals, cell-by-cell notebook audits, and execution fixtures that guided the refactoring process:

* [`notebook_audit.md`](notebook_audit.md): Cell-by-cell audit of [`Notebooks/GradientVanishingAnalysis.ipynb`](../Notebooks/GradientVanishingAnalysis.ipynb), mapping notebook state, execution flow, dependencies, and potential reproducibility risks.
* [`architecture_proposal.md`](architecture_proposal.md): Structural blueprint dividing notebook code into modular components within the `gradient_vanishing` package.
* [`refactoring_spec.md`](refactoring_spec.md): Approved technical specification establishing data contracts, numerical tolerances, CLI interfaces, and behavior-preservation guardrails.
* [`ticket_plan.md`](ticket_plan.md): 14-ticket implementation roadmap outlining sequential milestones, dependencies, and definitions of done.
* [`baseline_fixtures.md`](baseline_fixtures.md): Methodology for extracting, verifying, and comparing immutable execution oracles from original notebook definitions.
* [`end_to_end_validation.md`](end_to_end_validation.md): Verification report covering isolated wheel installation, CLI execution, repeatability, and software workflow validation.

---

## 4. Agent Standards and Project Conventions

Operational rules and skill conventions for AI coding agents and human maintainers:

* [`agents/domain.md`](agents/domain.md): Single-context domain layout guidelines linking root [`CONTEXT.md`](../CONTEXT.md) and architectural decision records.
* [`agents/issue-tracker.md`](agents/issue-tracker.md): Local Markdown issue tracker conventions managing feature tasks and specifications under [`.scratch/`](../.scratch/).
* [`agents/triage-labels.md`](agents/triage-labels.md): Standard triage role strings and status labels used across project tickets.

---

## 5. Technical Followups and Known Observations

Targeted technical notes investigating specific behaviors and theoretical refinements identified during refactoring:

* [`followups/01-gru-gate-terminology.md`](followups/01-gru-gate-terminology.md): Analysis of GRU gate conventions, clarifying the sign and parameter assignment of Update Gate bias `W_z.bias` vs Reset Gate `W_r`.
* [`followups/02-evaluate-mode.md`](followups/02-evaluate-mode.md): Analysis of the historical `evaluate()` helper's side-effect on model training mode.
* [`followups/03-learning-curve-uncertainty.md`](followups/03-learning-curve-uncertainty.md): Recommendations for plotting standard-deviation uncertainty bands on validation learning curves.

---

## Related Root Documents

* [`../README.md`](../README.md): Primary project overview, quick-start guide, and output interpretation.
* [`../CONTEXT.md`](../CONTEXT.md): Core domain glossary and architectural definitions.
* [`../AGENTS.md`](../AGENTS.md): Root instructions for agent skills and conventions.
* [`../PROJECT_STRUCTURE.md`](../PROJECT_STRUCTURE.md): Comprehensive directory map and Git status review.
