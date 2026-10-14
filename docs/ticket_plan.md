# Gradient Vanishing refactor: ticket plan for review

Status: **published as 14 local Markdown issues**. Sources: [architecture proposal](architecture_proposal.md), [refactoring specification](refactoring_spec.md), and the untouched source notebook. The issue files live under `.scratch/gradient-vanishing-refactor/issues/`, numbered in the order below.

## Shared guardrails

Every ticket must leave the source notebook unchanged, with SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`. A ticket may read or execute its code to capture evidence but may not edit, rename, or overwrite it. Preserve the documented algorithm, RNG order, full BPTT, custom GRU convention, final-model evaluation, and known evaluation-mode side effect during the initial extraction. Keep the three documented bug fixes as separate follow-up work. Saved notebook outputs are reference observations, not independently reproduced results. Baseline fixtures and their environment record must be captured before extraction. Provenance begins with the single-run training result in ticket 05 and is extended by tickets 06–10; ticket 12 standardizes it across commands.

The proposed work has **14 tickets**. Tickets 01–13 use small or reduced runs for verification. Ticket 14 alone performs the expensive notebook-equivalent experiment matrix and may report that empirical reproduction remains unverified if the approved hardware budget cannot support it. A completed ticket must provide its stated evidence and leave all earlier lightweight checks passing.

| Ticket | Title | Blocked by | Verification tier |
| --- | --- | --- | --- |
| 01 | Installable project and validated experiment configuration | None | Lightweight |
| 02 | Capture immutable baseline fixtures | 01 | Lightweight CPU |
| 03 | Reproduce the first-element memorization data | 02 | Lightweight |
| 04 | Extract recurrent models and gate initialization | 02 | Lightweight |
| 05 | Reproduce one seeded training and evaluation run | 03, 04 | Reduced training |
| 06 | Reproduce hidden-gradient diagnostics and full BPTT checks | 05 | Reduced backward runs |
| 07 | Reproduce one Optuna study | 05 | Reduced trials |
| 08 | Reproduce the default-versus-tuned main comparison | 06, 07 | Reduced grid |
| 09 | Reproduce the gate-initialization ablation | 06 | Reduced ablation |
| 10 | Reproduce the numeric retention-bias sweep | 06 | Reduced sweep |
| 11 | Regenerate tables and figures from saved results | 08, 09, 10 | Artifact fixtures |
| 12 | Standardize configured commands and run manifests | 11 | Reduced command runs |
| 13 | Validate the project end to end and document use | 12 | Reduced end-to-end |
| 14 | Run and report the full experiment matrix | 13 | **Expensive empirical run** |

Tickets 03 and 04 can proceed independently once 02 is complete. Ticket 07 can proceed after 05 while 06 is in progress. Tickets 09 and 10 can proceed independently after 06. No ticket depends on another merely to enforce a preferred order.

## 01 — Installable project and validated experiment configuration

- **Objective:** A researcher can install the project in a pinned reference environment and validate a versioned JSON configuration without executing the notebook.
- **Scope:** Package metadata, a committed resolved dependency manifest for the reference CPU environment, a default JSON configuration matching notebook constants, validation and override precedence, and a small configuration check. Document CUDA as a separate environment if supported.
- **Dependencies:** None; can start immediately.
- **Files or components affected:** Package metadata, dependency manifest, configuration loader, default configuration, configuration tests, installation instructions.
- **Acceptance criteria:** Clean install/import succeeds; unknown keys and invalid combinations fail clearly; command override takes precedence over configuration value and checked-in default; baseline values include the three sequence lengths, model names, seeds, update budgets, data sizes, search space, and gate mode `one`.
- **Required tests:** Load/default equality, accepted override, unknown-key rejection, invalid-type/range rejection, clean import without training or plotting.
- **Behavior-preservation checks:** No source notebook edit or runtime `pip install`; configuration defaults reproduce its constants. Deterministic CUDA setup must occur before Torch import when requested.
- **Definition of Done:** Environment and configuration checks pass in a clean install; notebook hash is unchanged; dependency versions are recorded. No model implementation is extracted yet.

## 02 — Capture immutable baseline fixtures

- **Objective:** A reviewer has baseline numerical evidence before any extraction can change the implementation.
- **Scope:** Run small, deterministic CPU cases directly from the untouched notebook definitions in the pinned environment; record initial parameters, one-step predictions, short loss traces, validation/test MSE, and hidden-gradient norms. Record Python and dependency versions, CPU/device details, deterministic settings, source revision, notebook hash, invocation, cell provenance, and fixture checksums before extraction. Predeclare comparison tolerances.
- **Dependencies:** 01.
- **Files or components affected:** Baseline fixture generator or capture script, fixture data, fixture metadata, fixture verification tests.
- **Acceptance criteria:** Fixtures cover Vanilla RNN, LSTM, and custom GRU at short and longer sequence lengths; parameter arrays, seed/config metadata, complete baseline environment, and fixture provenance are stored before extraction; saved notebook output cells are labeled historical observations and are not used as proof of reproduction.
- **Required tests:** Re-run the capture on the same pinned CPU environment and compare initial parameters exactly and computed float32/float64 quantities at the tolerances in the approved specification; verify fixture source hash and immutability policy.
- **Behavior-preservation checks:** Execute definitions without changing notebook cells or RNG call order. Capture the original evaluation-mode side effect and full BPTT behavior rather than correcting them.
- **Definition of Done:** A reviewer can reproduce and inspect the small-case fixtures and baseline environment; notebook SHA-256 still matches. This ticket completes before any implementation extraction in tickets 03–06.

## 03 — Reproduce the first-element memorization data

- **Objective:** A researcher can request fresh training batches and fixed validation/test sets with the notebook's exact task semantics.
- **Scope:** Extract synthetic sequence generation, fixed-set generators, per-sequence-length caching, and theoretical memoryless baseline. Keep CPU sampling before device transfer.
- **Dependencies:** 02.
- **Files or components affected:** Data module, data-contract tests, small data demonstration.
- **Acceptance criteria:** Inputs have shape `(T,B,1)` with only step zero sampled from `Uniform[-1,1]`; targets equal that first value and have shape `(B,1)`; validation/test sizes and seeds match the default config; same fixed-set seed pairs first values across lengths.
- **Required tests:** Shape, range, zero tail, target equality, split repeatability and separation, per-length cache, seeded small-case comparison to baseline fixture.
- **Behavior-preservation checks:** Training draws from the global Torch RNG after run seeding; fixed sets use separate CPU generators and do not consume that stream. No new data loader or dataset changes sampling order.
- **Definition of Done:** Data module can be called alone, tests pass, and fixture comparison shows no data-semantic change.

## 04 — Extract recurrent models and gate initialization

- **Objective:** A researcher can instantiate each hand-written model and obtain the same predictions and initial parameters as the notebook.
- **Scope:** Extract Vanilla RNN, LSTM, custom GRU, hidden-state tracking, weight initialization, retention-bias modes, and model construction. Keep initialization in the model area; do not create a registry or extra abstraction hierarchy.
- **Dependencies:** 02. It can proceed alongside 03.
- **Files or components affected:** Models module, initialization and equation tests, baseline model fixture comparisons.
- **Acceptance criteria:** Sequence-first input produces a scalar per batch item; Vanilla RNN, LSTM, and GRU one-step equations match independent calculations; custom GRU uses `(1-z)` for old state and `z` for candidate; `W_z` is the Update Gate linear layer and only `W_z.bias` receives the GRU retention-bias assignment; `W_r` is the Reset Gate layer.
- **Required tests:** Independent equation cases; zero initial states; output shape; exact seeded initial parameter arrays; orthogonal/Xavier scope over all two-dimensional weights; unchanged biases where specified; zero/one/chrono/numeric gate signs, including LSTM chrono input-gate pairing.
- **Behavior-preservation checks:** Construct and initialize on CPU before device transfer; preserve constructor RNG order, manual gate equations, custom GRU sign, and tracked hidden tensors. Detached LSTM cell snapshots must not detach the live recurrent cell state.
- **Definition of Done:** Every model builds and predicts in isolation, all equation/initialization tests pass, and notebook hash is unchanged.

## 05 — Reproduce one seeded training and evaluation run

- **Objective:** A researcher can train one selected model/length/configuration and obtain a history plus final-model validation/test MSE.
- **Scope:** Extract run seeding, fresh-batch training, Adam/MSE, gradient clipping, validation schedule, pruning hooks, and evaluation. Expose the final model state and a minimal per-run metadata record containing effective configuration, random seed, dependency versions, effective hyperparameters, device, and notebook hash. Do not add best-checkpoint selection or a separate checkpoint manager because the notebook has neither.
- **Dependencies:** 03 and 04.
- **Files or components affected:** Training module, single-run result and metadata record, training-contract tests.
- **Acceptance criteria:** Each update zeros gradients, generates a fresh batch, computes loss, backpropagates, captures selected gradients, clips parameter gradients at norm 5, steps Adam, then records pre-step train loss and scheduled post-step validation loss. The trial hook reports validation loss and can prune. Final MSE uses final weights. Every run returns or saves minimal metadata with configuration, seed, dependency versions, and effective hyperparameters.
- **Required tests:** Seeded short-run fixture comparison, optimizer-update count, validation at updates divisible by ten and final update, fresh-batch behavior, clip order, final-model evaluation, no-grad evaluation, trial report/prune behavior, metadata completeness and reload.
- **Behavior-preservation checks:** Keep the global training RNG stream and exact seed/init/data ordering. Preserve `evaluate`'s current unconditional return to training mode during extraction; its correction remains a separate follow-up task.
- **Definition of Done:** One reduced seeded training run matches the baseline fixture within declared tolerances and emits its minimal provenance record; no checkpoint-selection behavior has been added.

## 06 — Reproduce hidden-gradient diagnostics and full BPTT checks

- **Objective:** A researcher can inspect hidden-state gradient flow at initialization and during training with the notebook's metric.
- **Scope:** Extract gradient-norm reading, float64 ten-seed initial-gradient curves, geometric aggregation, decay-factor fit, and selected pre-clipping training captures. Return numeric arrays and the seed list, model/configuration, sequence length, dtype, device, and fit/capture settings needed to reproduce them.
- **Dependencies:** 05.
- **Files or components affected:** Experiment analysis functions, gradient result records, gradient/BPTT tests.
- **Acceptance criteria:** Intermediate hidden states remain connected across the full sequence; `dL/dh_t` uses mean batch L2 norm; initial curves divide by last-step norm; decay uses the notebook fit window; default capture indices are 0, 50, 150, 299, with valid reduced-run indices supplied only by smoke config. Saved metadata identifies the seeds, dtype, model/configuration, sequence length, and fit/capture settings.
- **Required tests:** Independent short-sequence gradient calculation, first-step gradient connection, norm ordering/length, float64 ratio and geometric mean, decay fit, capture before clipping and optimizer step, zero/non-finite ratio reporting.
- **Behavior-preservation checks:** Do not truncate BPTT, detach live hidden/cell states, substitute parameter gradients, or claim hidden-only metrics measure all LSTM cell paths. Preserve notebook clipping and normalization rules.
- **Definition of Done:** Reduced gradient run matches baseline small-case fixtures, emits arrays needed for both plots, and passes full-BPTT tests.

## 07 — Reproduce one Optuna study

- **Objective:** A researcher can tune a model at one sequence length using the notebook's selection rule and inspect the selected parameters.
- **Scope:** One study over the shared learning-rate, hidden-size, and weight-initialization search space; TPE sampler, median pruner, trial-number-derived seed, 150-update default tuning budget, and last validation MSE objective. Save study and trial metadata: model, sequence length, trial seeds, search space, sampler/pruner settings, budgets, trial outcomes, and selected parameters. Support a reduced trial/update budget for tests.
- **Dependencies:** 05. It can proceed while 06 is built.
- **Files or components affected:** Experiment orchestration, study result record, tuning tests.
- **Acceptance criteria:** Default study uses 20 trials, TPE seed 42, five startup trials and 50 warm-up steps, trial seed `1000 + trial number`; gate mode stays `one` and is not tuned; selected config contains only the notebook's three searched parameters. Metadata can reproduce the study settings and identify pruned/failed/completed trials.
- **Required tests:** Search-space limits and categories, per-trial seed, pruning/reporting sequence, objective equals final validation rather than minimum history value, reduced study result serialization, test-split non-access.
- **Behavior-preservation checks:** Do not change study order, tune gate bias, optimize test MSE, average trial seeds, or select best checkpoint. A full nine-study run waits for ticket 14.
- **Definition of Done:** A reduced study completes and exposes its trial/selection evidence; notebook hash and light regression checks pass.

## 08 — Reproduce the default-versus-tuned main comparison

- **Objective:** A researcher can compare Vanilla RNN, LSTM, and GRU at configured sequence lengths with default and tuned hyperparameters over the same seeds.
- **Scope:** Orchestrate the three-length, three-model, two-condition, five-seed grid; invoke tuning per `(length, model)`; collect final test MSE and histories, including selected training gradients. Save per-run model, length, condition, seed, effective parameters, validation/test split seeds, and tuning-study reference. Use reduced config in ticket tests.
- **Dependencies:** 06 and 07.
- **Files or components affected:** Main experiment orchestration, raw comparison result record, grid tests.
- **Acceptance criteria:** Default config schedules nine studies and 90 final runs; both conditions use retention-bias mode `one`; each run uses final weights and the fixed test set for its length; raw per-seed values, selected tuned parameters, and per-run reproduction metadata are available for reporting.
- **Required tests:** Reduced grid cardinality and ordering, exact seeds, same fixed test set per length, test exclusion from tuning, final test MSE, captured gradient indices, sample mean/std (`ddof=1`).
- **Behavior-preservation checks:** Keep the 150-versus-300-update distinction and default-versus-tuned semantics. Do not claim saved notebook numbers are reproduced by reduced runs.
- **Definition of Done:** A reduced grid yields inspectable raw results and passes the baseline-compatible orchestration tests; full grid remains ticket 14.

## 09 — Reproduce the gate-initialization ablation

- **Objective:** A researcher can compare zero, one, and chrono retention bias against the Vanilla RNN reference at `T=100`.
- **Scope:** Run the seven notebook conditions with default hyperparameters, five seeds, final test MSE, first checked validation MSE below `0.01`, and ten-seed initial-gradient decay. Save each condition's model, gate mode, length, seeds, effective config, threshold, and gradient settings. Allow smaller validation runs.
- **Dependencies:** 06.
- **Files or components affected:** Ablation orchestration, raw ablation results, ablation tests.
- **Acceptance criteria:** Default config schedules 35 training runs; Vanilla RNN has no gate intervention; LSTM and GRU each use zero/one/chrono; threshold speed is based only on validation checks and reports success count; decay uses the same initial-gradient metric. Saved metadata identifies every condition and its run/measurement settings.
- **Required tests:** Condition matrix, gate-parameter signs, threshold resolution and non-achievement, mean/std formatting inputs, test isolation, reduced ablation result schema.
- **Behavior-preservation checks:** `W_z.bias` is the GRU intervention, not `W_r`; chrono's RNG draw and LSTM input-bias pairing stay intact; no gate bias enters Optuna.
- **Definition of Done:** A reduced ablation runs independently of the main grid and produces raw metrics for later table generation.

## 10 — Reproduce the numeric retention-bias sweep

- **Objective:** A researcher can measure how initial gradient decay varies with the abstract retention bias for LSTM and custom GRU.
- **Scope:** Sweep 17 values from `-4` through `4`, five initialization seeds per point, with theoretical sigmoid retention values and measured decay factors. Return numeric plot inputs plus model, length, exact bias grid, seeds, dtype, initialization, and fit settings.
- **Dependencies:** 06. It can proceed alongside 09.
- **Files or components affected:** Sweep orchestration, raw sweep results, sweep tests.
- **Acceptance criteria:** Default sweep uses `T=100`, both gated models, 17 evenly spaced biases, and five seed curves per bias; LSTM uses `+b` at forget bias, GRU uses `-b` at Update Gate bias; results include measured decay, theoretical retention, and enough metadata to repeat each measurement.
- **Required tests:** Sweep grid/count, endpoint and spacing checks, gate sign, reduced sweep reproducibility, decay calculation, no training/test-set access.
- **Behavior-preservation checks:** Keep model construction and RNG order per bias/seed, float64 gradient measurement, and the notebook's custom GRU equation. Do not reinterpret the sweep as a Reset Gate intervention.
- **Definition of Done:** A reduced sweep independently yields numeric arrays needed for the reference plot.

## 11 — Regenerate tables and figures from saved results

- **Objective:** A researcher can view the main comparison, gradient diagnostics, ablation, and bias sweep without rerunning models.
- **Scope:** Report the nine-row comparison table, log-scale mean validation curves, initial-gradient semilog curves, tuned seed-0 `T=100` heatmap, ablation table, and measured-versus-theoretical sweep plot from raw results.
- **Dependencies:** 08, 09, and 10.
- **Files or components affected:** Reporting module, artifact schema validation, reporting tests.
- **Acceptance criteria:** All six notebook report types can be regenerated from persisted numeric inputs; mean/std uses `ddof=1`; the theoretical `1/3` line is distinguished from finite-sample test MSE; heatmap normalization and display clipping match the notebook.
- **Required tests:** Known synthetic result fixtures for table values, curve point alignment, log scaling, heatmap matrix and clip limits, sweep reference line, report regeneration without RNG consumption or training.
- **Behavior-preservation checks:** Keep the main learning curve mean-only during extraction, despite its misleading source comment; the uncertainty-band fix is a separate follow-up. Do not alter metrics merely to simplify plotting.
- **Definition of Done:** Reports regenerate from stored fixture results and match the specified visual/numeric semantics; raw result files remain unchanged.

## 12 — Standardize configured commands and run manifests

- **Objective:** A researcher can invoke `main`, `gradients`, `ablation`, `sweep`, and `report` from a clean install and identify exactly what each run produced through one consistent manifest.
- **Scope:** Thin command entry points, configuration and output selection, and a standardized schema that assembles the provenance already emitted by tickets 05–10. Finalize run status, result persistence, and report regeneration. The training heatmap uses main-run gradient histories; initial-gradient analysis, ablation, and sweep can run independently.
- **Dependencies:** 11.
- **Files or components affected:** CLI adapter, run manifest and raw-result serialization, command tests, usage examples.
- **Acceptance criteria:** Each command accepts validated config and an output location; manifest consistently aggregates existing per-run provenance and records source revision, notebook hash, effective config, environment/device/determinism, seeds, selected parameters, command, and completion status; `report` reads existing results without training.
- **Required tests:** Reduced command invocations, invalid-config exit, incomplete/pruned run status, artifact reload and schema check, report-only command with training disabled, no import-time side effects.
- **Behavior-preservation checks:** Serialization and reporting do not consume RNG or change training order. CUDA determinism setting is applied before Torch import; unsupported deterministic operations surface errors.
- **Definition of Done:** All five commands run on reduced inputs and produce inspectable, reloadable artifacts; no full empirical reproduction claim is made.

## 13 — Validate the project end to end and document use

- **Objective:** A new researcher can install, run, inspect, and rerun a reduced experiment using only documented commands.
- **Scope:** Integration tests, pinned-CPU baseline comparisons, fresh-environment instructions, result interpretation, known-issue links, and a reduced end-to-end run through tune → train → evaluate → save → report.
- **Dependencies:** 12.
- **Files or components affected:** Integration tests, research usage documentation, reproducibility checklist, reduced-run artifacts.
- **Acceptance criteria:** Reduced pipeline completes twice in the pinned CPU environment within declared tolerances; raw results and manifests agree; test data never reaches tuning; reports regenerate without retraining; every cell in the approved spec's coverage map remains accounted for.
- **Required tests:** Clean install, two seeded reduced runs, split-isolation check, source hash check, baseline fixture comparison, artifact round-trip, CLI/report smoke test. Document any environment-specific discrepancy.
- **Behavior-preservation checks:** Known bugs remain separately documented; no best-checkpoint selection or new algorithm enters integration. Treat saved notebook outputs only as unverified reference observations.
- **Definition of Done:** Lightweight acceptance gates pass and documentation lets another researcher repeat them. The result is called structurally validated, not yet empirically reproduced.

## 14 — Run and report the full experiment matrix

- **Objective:** A reviewer can assess whether the modular project reproduces the notebook's empirical conclusions under a recorded environment.
- **Scope:** Full nine-study tuning matrix, 90 final main runs, 35 ablation training runs, initial-gradient analyses, numeric bias sweep, all reports, and a final reproducibility report. This is the sole ticket whose default path requires extensive computation.
- **Dependencies:** 13.
- **Files or components affected:** Full-run artifacts, final report, discrepancy analysis; implementation modules change only if a verified regression is found.
- **Acceptance criteria:** The manifest distinguishes completed, pruned, failed, and unrun work; per-seed raw metrics and generated figures are available; the report compares new results with notebook saved outputs while stating that those outputs were not independently verified; any discrepancy is classified by code, environment, stochastic variation, or stale source output.
- **Required tests:** Re-run lightweight gates first, then verify matrix cardinalities, output completeness, summary arithmetic, test-set isolation, notebook hash, and artifact regeneration. Any corrected implementation must rerun its relevant lightweight and full checks.
- **Behavior-preservation checks:** No silent algorithm, gate, RNG, gradient, training, or evaluation changes are used to make numbers match. Cross-device bitwise equality is not required; only the pinned reference environment supports the direct numerical comparison.
- **Definition of Done:** A full empirical reproduction report is complete if the approved compute budget permits the run. Otherwise the ticket records the concrete hardware/runtime limit and marks empirical reproduction **unverified**, without presenting reduced results as full reproduction.

## Publication

The approved 14-ticket breakdown and blocking edges were published one file per ticket in the configured local Markdown tracker. Each issue links to the approved specification and records its own objective, scope, dependencies, affected components, acceptance criteria, tests, preservation checks, and Definition of Done. Ticket 01 is the initial unblocked frontier; no implementation has begun.
