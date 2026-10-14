# 02: Capture immutable baseline fixtures

Status: resolved
Type: task
Blocked by: 01

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)  
**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)  
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## Objective

A reviewer has baseline numerical evidence before any extraction can change the implementation.

## Scope

Run small, deterministic CPU cases directly from the untouched notebook definitions in the pinned environment; record initial parameters, one-step predictions, short loss traces, validation/test MSE, and hidden-gradient norms. Record Python and dependency versions, CPU/device details, deterministic settings, source revision, notebook hash, invocation, cell provenance, and fixture checksums before extraction. Predeclare comparison tolerances.

## Dependencies

01.

## Files or components affected

Baseline fixture generator or capture script, fixture data, fixture metadata, fixture verification tests.

## Acceptance criteria

- [x] Fixtures cover Vanilla RNN, LSTM, and custom GRU at short and longer sequence lengths; parameter arrays, seed/config metadata, complete baseline environment, and fixture provenance are stored before extraction; saved notebook output cells are labeled historical observations and are not used as proof of reproduction.

## Required tests

Re-run the capture on the same pinned CPU environment and compare initial parameters exactly and computed float32/float64 quantities at the tolerances in the approved specification; verify fixture source hash and immutability policy.

## Behavior-preservation checks

Execute definitions without changing notebook cells or RNG call order. Capture the original evaluation-mode side effect and full BPTT behavior rather than correcting them.

## Definition of Done

- [x] A reviewer can reproduce and inspect the small-case fixtures and baseline environment; notebook SHA-256 still matches. This ticket completes before any implementation extraction in tickets 03–06.

## Answer

The unchanged notebook and approved design documents were committed in source snapshot `72418d9241b81e8ffefc73c917eab6663413cb10` before fixture capture. The [capture script](../../../scripts/capture_notebook_baseline.py) executes original notebook cells 3, 5, 7, 9, 11, and 13 in order in the pinned Python 3.13/Torch 2.11 CPU environment. The [immutable fixture](../../../tests/fixtures/notebook_baseline.json) holds 22 directly executed cases: one synthetic-data case, 12 model/gradient cases across three models and short/long sequences, six gate-initialization cases, and three short seeded training/evaluation cases. [Documentation](../../../docs/baseline_fixtures.md) records the scope and limitations.

Verification: a fresh capture matched exact initial parameter arrays and computed float32/float64 observations within the approved tolerances. Tests check provenance, schema, environment, source and fixture hashes, full BPTT observations, the evaluation-mode side effect, and refusal to overwrite a fixture. `python -m pytest -q` passed 21 tests in the pinned environment; `pip check` and Python compilation passed. The notebook SHA-256 remains `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`. Saved notebook outputs were not treated as verified results. The three-update probes do not establish the full experimental matrix. Code review findings about exact gate-parameter comparison and environment checks were fixed.
