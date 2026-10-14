# Modular research project specification

Status: **approved for ticket planning; tickets published locally**. This is a design deliverable; implementation is pending. Its baseline is the [notebook audit](notebook_audit.md), and its proposed module layout is in the [architecture proposal](architecture_proposal.md). The source notebook remains byte-for-byte unchanged.

## Problem Statement

The first-element memorization experiment exists only as an order-dependent notebook. Data generation, recurrent equations, initialization, training, tuning, metrics, and plots share mutable kernel state. A researcher cannot reliably run one stage, reproduce a reported result in a clean environment, or test whether a structural change preserved the original experiment. Dependencies and execution environment are unpinned, and saved outputs do not establish a verified clean run.

## Solution

Provide a small, installable Python research project with explicit configuration and commands for the main model comparison, initial-gradient analysis, gate ablation, and numeric bias sweep. Each run records its effective configuration, environment, seeds, selected hyperparameters, raw results, and generated reports. A researcher can rerun a stage or regenerate reports from saved results without notebook state. The first refactor preserves the original algorithms and experimental semantics; known bugs have separate follow-up tasks.

## User Stories

1. As a researcher, I want to install one documented environment, so that a clean machine can run the experiment.
2. As a researcher, I want the baseline notebook preserved unchanged, so that I can compare the project against its source.
3. As a researcher, I want a named configuration for every experiment setting, so that I can see exactly what ran.
4. As a researcher, I want to override a small set of run controls without editing source, so that I can run a cheap smoke experiment before the full grid.
5. As a researcher, I want deterministic seed handling, so that repeated runs on a supported environment are comparable.
6. As a researcher, I want the first-element memorization task isolated behind one data interface, so that its shapes, distribution, and split rules can be checked.
7. As a researcher, I want the three recurrent equations visible in one model area, so that architectural comparisons remain inspectable.
8. As a researcher, I want the custom GRU's Update Gate and Reset Gate named correctly, so that the retention-bias intervention is unambiguous.
9. As a researcher, I want weight and gate initialization separated from the model forward passes, so that bias conditions can be compared without changing equations.
10. As a researcher, I want one single-run training operation, so that tuning, main runs, and ablations follow the same update and validation rules.
11. As a researcher, I want each Optuna study to expose its search space, budget, seed, pruner, and selected parameters, so that model selection can be reproduced.
12. As a researcher, I want the default and tuned conditions run over the same five seeds and fixed test set, so that the reported comparison matches the notebook.
13. As a researcher, I want test data excluded from hyperparameter selection, so that the test metric remains an evaluation result.
14. As a researcher, I want per-seed losses and gradient measurements saved, so that aggregate tables and figures can be inspected and regenerated.
15. As a researcher, I want the initial-gradient and training-gradient analyses preserved, so that gradient-flow claims can be checked against the same metric.
16. As a researcher, I want the zero/one/chrono and numeric retention-bias studies runnable independently, so that I can investigate gate initialization without rerunning the main grid.
17. As a reviewer, I want a run manifest and explicit completion status, so that I can distinguish a full experiment from a partial or pruned run.
18. As a reviewer, I want scientific compatibility tests against the untouched baseline, so that a changed result is detected before conclusions are updated.
19. As a maintainer, I want a small number of clear modules, so that new research questions can reuse the core without a framework or hidden global state.
20. As a maintainer, I want existing bugs documented and fixed in separate changes, so that behavior changes can be attributed and regression-tested.

## Implementation Decisions

- Use a small package with areas for configuration, synthetic data, recurrent models and initialization, single-run training, experiment orchestration and analysis, and reporting. A thin command entry point selects stages. Do not add a plugin system, database, model registry, or service.
- Use a versioned JSON experiment configuration, parsed and validated with the standard library, for sequence lengths, model names, seeds, update budgets, batch and evaluation sizes, learning rate and hidden size, initialization mode, device, and output location. The checked-in default configuration reproduces the notebook. Limited command overrides affect only the requested run; the effective values are saved. Define precedence as command override over configuration value over checked-in default; reject unknown keys and invalid combinations rather than silently substituting values.
- Keep plain functions for data, training, analysis, and orchestration, with the three recurrent model classes and their initialization grouped together. Use small typed result records where they make saved data or interfaces clearer. Module import must not train, install packages, or render plots.
- Preserve sequence-first `(T, B, 1)` inputs, `(B, 1)` targets, a scalar sampled uniformly from `[-1,1]` only at the first step, fresh training batches, fixed validation/test generators, and their baseline seeds. Data generation remains on CPU before transfer to the selected device. Keep the global Torch RNG stream for training batches; an explicit per-batch generator would alter baseline samples.
- Preserve the hand-written Vanilla RNN, LSTM, and custom GRU equations. In this GRU, `z` blends candidate state in and `1-z` keeps old state. `W_z` is the Update Gate linear layer with weight and bias parameters; retention-bias initialization changes `W_z.bias`. `W_r` is the Reset Gate layer and is not the ablation target.
- Preserve initialization order and random-number consumption where observable: construct on CPU, initialize weights and gate parameters, then move to device. The abstract retention bias is positive for LSTM forget bias and negative for this GRU's Update Gate bias. Chrono also negates the LSTM input-gate bias.
- Preserve Adam, MSE, gradient clipping at norm 5, final-checkpoint evaluation, fixed validation cadence, 150 tuning updates, 300 final-run updates, and the trial report/prune sequence. Training loss is captured before each optimizer step; validation is measured after the step. The notebook calls each optimizer update an “epoch”; the project should call it an update in public interfaces while mapping saved notebook labels explicitly.
- Preserve one 20-trial study per model and sequence length: identical common-parameter search space, TPE sampler seed 42, median-pruner settings, and trial seed derived from trial number. Gate bias is outside tuning. The `default` condition means default hyperparameters while both main conditions use gate mode `one`.
- Keep test access outside the Optuna objective. Measure test MSE on the final model for each seed; report arithmetic mean and sample standard deviation. Keep the theoretical memoryless baseline `1/3` distinct from the finite test-set zero-predictor MSE.
- Preserve hidden-state gradient tracking, pre-clipping measurement times, float64 initial-gradient analysis, geometric aggregation, decay-fit window, float32 heatmap, and threshold-check cadence. Do not present the hidden-state metric as a measurement of every LSTM cell-state gradient path.
- Persist machine-readable raw per-seed results and a run manifest before producing human-readable tables and figures. Reporting reads persisted results and does not retrain or consume random numbers. Record incomplete runs explicitly. The manifest includes source revision, notebook hash, effective configuration, dependency versions, Python/Torch/CUDA and device details, deterministic settings, seeds, command, and completion status.
- Commit a resolved dependency manifest for one tested reference environment alongside package metadata. Record any separately supported CUDA environment and set `CUBLAS_WORKSPACE_CONFIG` before Torch import; deterministic-operation failures are reported rather than bypassed.
- Keep known defects separate from extraction: [GRU wording](followups/01-gru-gate-terminology.md), [evaluation-mode restoration](followups/02-evaluate-mode.md), and [learning-curve uncertainty display](followups/03-learning-curve-uncertainty.md). If a defect blocks faithful extraction, surface the conflict before altering behavior.

## Notebook coverage map

Cell numbers are zero-based JSON indices. The mapping identifies the destination and a verification point for every notebook cell. Explanatory Markdown becomes research documentation rather than executable modules.

| Cells | Baseline responsibility | Proposed home | Verification |
| --- | --- | --- | --- |
| 0 | Install Optuna in kernel | Package metadata and resolved environment | Clean install and import without runtime package installation |
| 1–2 | Research objective, baseline, setup | Research documentation and configuration description | Documentation agrees with task distribution and population baseline |
| 3 | Imports, device, deterministic setting, seeding | Command bootstrap and configuration | Environment manifest, repeatability, CUDA setting before Torch import |
| 4–5 | Constants and main gate mode | Validated default configuration | Exact baseline defaults and override precedence |
| 6–7 | Data generator, fixed sets, cache, baseline | Data module | Shape, distribution, split seeds, cross-length pairing, cache behavior |
| 8–9 | Equations, three models, hidden tracking | Models module | Independent one-step equations, full-sequence gradients, output shape |
| 10–11 | Weight/gate initialization and factory | Models module | Weight scope, gate signs, chrono pairing, CPU-before-device order |
| 12–13 | Evaluation and seeded training | Training module | Update order, validation cadence, clipping, final model, mode side effect |
| 14–15 | Search description and Optuna study | Experiments module | Search space, trial seed, objective, sampler/pruner, 20-trial budget |
| 16–17 | Main tune/default/tuned/five-seed grid | Experiments module | Run matrix, test-only final evaluation, recorded histories |
| 18–19 | Summary table and sample spread | Reporting module | Nine rows, mean and `ddof=1` standard deviation |
| 20–21 | Mean validation learning curves | Reporting module | Log-scale mean curves and theoretical baseline; known missing band stays separate |
| 22–23 | Float64 initial-gradient curves and decay | Experiments calculation + reporting plot | Ratio, geometric mean, fit window, ten seeds |
| 24–25 | Tuned seed-0 gradient heatmap | Reporting module from main-run gradient histories | Four capture points, normalization and display clipping |
| 26–27 | Zero/one/chrono ablation and table | Experiments calculation + reporting table | Seven conditions, five seeds, threshold checks, decay summary |
| 28 | Numeric retention-bias sweep | Experiments calculation + reporting plot | 17 values, five seeds each, measured decay and reference line |

## Scientific preservation contract

- **Vanilla RNN:** each step applies `tanh` to the sum of the input projection and recurrent hidden projection. Its recurrent projection has no bias; the input projection does. Prediction uses a linear projection of the final hidden state.
- **LSTM:** input, forget, candidate, and output projections consume concatenated input and previous hidden state. The cell update adds forgotten previous cell state and gated candidate; hidden state is output gate times `tanh` of the new cell state. Initial hidden and cell states are zeros. The stored cell-state snapshots are detached for inspection, while the live cell state in the recurrence remains connected to autograd.
- **Custom GRU:** `W_z` produces the Update Gate preactivation from input and previous hidden state; `W_r` produces the Reset Gate preactivation. The reset gate multiplies previous hidden state only in the candidate path. The new hidden state weights old state by `1-z` and candidate by `z`. Keep this blending convention and the negative sign of the retention bias applied to `W_z.bias`.
- **Initialization scope:** orthogonal and Xavier modes replace every two-dimensional weight parameter, including gate/input, recurrent, candidate, and output projections; they do not reinitialize biases. `default` retains each linear layer's constructor initialization. Gate initialization then changes only its named bias parameters; `one` changes the LSTM forget bias or GRU Update Gate bias, while chrono also changes the LSTM input-gate bias.
- **Full BPTT:** every time step remains in the autograd graph until the final-step MSE calls `backward()`. There is no hidden/cell-state detachment or truncated-BPTT window in training. The hidden tensors call `retain_grad()` when gradients are enabled; their batch-mean L2 gradient norms are read after backward and before clipping.
- **Update and evaluation order:** reseed, construct/init/move model, create Adam optimizer, obtain fixed validation set, then per update: zero gradients, generate batch, forward/MSE, backward, capture selected hidden gradients, clip parameter gradients, optimizer step, append pre-step training loss, and evaluate at the stated post-step schedule. `evaluate` uses no-grad and currently leaves the model in training mode; preserve this side effect in the first extraction and correct it separately.
- **Visualization semantics:** main learning curves plot mean validation MSE on a log y-axis and a horizontal `1/3` line; the missing standard-deviation band is a documented defect. Initial-gradient curves use semilog display. The heatmap uses log10 of normalized gradient, clipped at `10^-10`; the bias sweep compares measured decay with the theoretical `y=x` reference. Saved numeric arrays must support regenerating every figure.
- **Numerical edge cases:** the initial-gradient calculation divides by the final hidden-gradient norm, then clips values before taking logs; the heatmap adds a small denominator guard and clips its color values. Keep these baseline calculations during extraction, report zero/non-finite ratios explicitly, and handle any correction in a separate reviewed change.

## Testing Decisions

The primary test seam is a configured experiment invocation and its externally visible saved manifest, raw results, table, and figures. Focused tests cover mathematical behavior and data contracts that a full run would hide or make expensive. Test observable values and invariants rather than internal call counts, except where update count or validation cadence is itself part of the experiment. There are no existing tests in this repo; the untouched notebook's code and saved outputs provide the available prior art, with saved outputs treated as unverified reference observations.

- **Data contract tests:** shape, target equality, zero tail, first-value range, fixed-set repeatability, validation/test separation, and same-seed cross-length pairing.
- **Model and initialization tests:** independent one-step calculations for each recurrent equation; final output shape; gradient-history order and length; verify gradients reach the first hidden state across the full sequence; both parameters of `W_z` identified separately; `W_z.bias` sign and `W_r` untouched by retention-bias initialization; chrono sign pairing in LSTM.
- **Seeded baseline comparisons:** capture small-case initial parameters, predictions, a short loss trace, final validation/test MSE, and hidden-gradient norms from code extracted directly from the untouched notebook in one pinned CPU environment. For the same pinned environment, compare parameters exactly and use predeclared float32 tolerance (`rtol=1e-5`, `atol=1e-6`) and float64 tolerance (`rtol=1e-8`, `atol=1e-10`) for computed values. The baseline fixture is recorded before module extraction and is not regenerated merely to make a failing test pass.
- **Single-run behavior tests:** final-checkpoint evaluation, fresh-batch generation, gradient capture before clipping, validation checks at baseline update indices, sample standard deviation, threshold timing, and pruning/reporting semantics. Preserve the current mode change by `evaluate` during extraction and cover its correction in the separate follow-up.
- **End-to-end smoke test:** run reduced budgets through tune → five-seed comparison → gradient/ablation report → persisted results → regenerated report. Use gradient capture points valid for the shortened budget without changing the default full-run points. Assert effective configuration, seeds, selected parameters, split usage, run status, and artifact consistency. The test must prove that test data never enters tuning and repeated pinned-CPU runs match within the declared tolerances.
- **Full-grid verification:** run the notebook-equivalent grid on the approved hardware, compare per-seed and aggregate results with baseline observations, and classify discrepancies as code change, environment difference, stochastic variation, or stale notebook output. Cross-device bitwise identity is not an acceptance condition.
- **Source protection:** compare the baseline notebook's SHA-256 before and after implementation. Its audited hash is `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`.

## Acceptance Criteria

1. A fresh installation from the committed resolved dependency manifest can invoke `main`, `gradients`, `ablation`, `sweep`, and `report` without executing notebook cells or installing packages at runtime. Python, Torch, Optuna, plotting versions, and device details are recorded for every run.
2. The checked-in JSON configuration reproduces all audited constants and rejects unknown or invalid values. A smoke override changes only the requested run and appears in its effective-config artifact.
3. Every notebook cell has the destination and verification point in the coverage map above; no executable analysis disappears during extraction.
4. Independent tests confirm Vanilla RNN/LSTM/GRU equations, full BPTT, initial zero states, gate roles and bias signs, weight-initialization scope, data shapes and splits, and hidden-gradient definitions. The custom GRU's `W_z` and `W_r` roles remain distinct.
5. A pinned-CPU baseline fixture captured before extraction matches the refactor's initial parameters exactly and its small-run predictions, losses, MSE, and gradients within the declared tolerances. A failing fixture is investigated rather than rewritten to match the new code.
6. A reduced configured run completes the tune → train → evaluate → save → report path. Its manifest reports effective config, seeds, package/device details, completion state, and selected parameters; regenerating tables/figures leaves raw results unchanged and does not train.
7. The default main grid schedules nine 20-trial studies and 90 final runs (three lengths × three models × two conditions × five seeds). The `T=100` ablation schedules 35 runs across seven conditions; the numeric sweep evaluates 17 biases with five initial-gradient seeds per model. Partial/pruned work is distinguishable from completed work.
8. Optuna sees validation data but not test data. Each final test MSE uses the final model state, aggregates use sample standard deviation, and the theoretical `1/3` baseline is identified separately from finite-sample test MSE.
9. The main summary, log-scale learning curves, initial-gradient curves, heatmap, ablation table, and numeric sweep plot can be regenerated from saved numeric results. The learning curve remains mean-only in the preservation phase, with its wording issue linked to a separate follow-up.
10. A full-grid run is required before claiming notebook-level empirical reproduction. If hardware or runtime prevents it, the project may be accepted as structurally refactored only; the manifest and report explicitly state that scientific reproduction remains unverified.
11. The source notebook SHA-256 stays `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`, and each documented bug remains separate from the behavior-preserving extraction unless its conflict is reviewed first.

## Out of Scope

Changing model equations, using library RNN layers, adding datasets or architectures, tuning gate bias, altering the trial objective or choosing best validation checkpoints, expanding statistical claims, fixing known bugs in the extraction change, providing an inference service, and editing or converting the source notebook.

## Further Notes

The [architecture proposal](architecture_proposal.md) gives the proposed module layout and data flow; the [audit](notebook_audit.md) records cell-level evidence and reproducibility risks. The selected test seam is a configured run and its saved artifacts, supplemented by focused mathematical checks. The issue tracker is configured as local Markdown with the default triage labels. The approved [ticket plan](ticket_plan.md) has been published as one local issue file per ticket; a tracker spec pointer links back to this document so it remains the authoritative source. No implementation has started.
