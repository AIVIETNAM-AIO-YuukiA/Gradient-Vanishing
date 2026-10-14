# 07: Reproduce one Optuna study

Status: resolved
Type: task
Blocked by: 05

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)
**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## Objective

A researcher can tune a model at one sequence length using the notebook's selection rule and inspect the selected parameters.

## Scope

One study over the shared learning-rate, hidden-size, and weight-initialization search space; TPE sampler, median pruner, trial-number-derived seed, 150-update default tuning budget, and last validation MSE objective. Save study and trial metadata: model, sequence length, trial seeds, search space, sampler/pruner settings, budgets, trial outcomes, and selected parameters. Support a reduced trial/update budget for tests.

## Dependencies

05. It can proceed while 06 is built.

## Files or components affected

Experiment orchestration, study result record, tuning tests.

## Acceptance criteria

- [x] Default study uses 20 trials, TPE seed 42, five startup trials and 50 warm-up steps, trial seed `1000 + trial number`; gate mode stays `one` and is not tuned; selected config contains only the notebook's three searched parameters. Metadata can reproduce the study settings and identify pruned/failed/completed trials.

## Required tests

Search-space limits and categories, per-trial seed, pruning/reporting sequence, objective equals final validation rather than minimum history value, reduced study result serialization, test-split non-access.

## Behavior-preservation checks

Do not change study order, tune gate bias, optimize test MSE, average trial seeds, or select best checkpoint. A full nine-study run waits for ticket 14.

## Definition of Done

- [x] A reduced study completes and exposes its trial/selection evidence; notebook hash and light regression checks pass.

## Answer

Implemented `src/gradient_vanishing/experiments.py` with the notebook's three-suggestion Optuna objective, TPE sampler, median pruner, trial seed rule, final-validation selection, and JSON study record. `docs/optuna_study.md` documents the workflow and reference boundary. A reduced three-trial Vanilla RNN study at `T=10` matched direct execution of the original notebook definitions for sampled parameters, objective values, trial statuses, intermediate reports, and best-trial selection. Focused tests also cover propagation, reproducibility, pruning, failure recording (including a partial record when no trial completes), and test-split exclusion. The full suite passed: 106 tests, 0 failures. A local wheel built, installed into a workspace target, and imported successfully; `pip check` found no broken requirements. The nine-study matrix was not run.
