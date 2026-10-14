# 04: Extract recurrent models and gate initialization

Status: resolved
Type: task
Blocked by: 02

**Approved specification:** [Refactoring specification](../../../docs/refactoring_spec.md)
**Architecture:** [Architecture proposal](../../../docs/architecture_proposal.md)
**Source notebook:** [GradientVanishingAnalysis.ipynb](../../../Notebooks/GradientVanishingAnalysis.ipynb)

**Shared preservation:** Keep the source notebook unchanged at SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`; do not silently fix documented bugs; treat its saved outputs as unverified observations.

## What to build

A researcher can instantiate each hand-written model and obtain the same predictions and initial parameters as the notebook.

## Objective

A researcher can instantiate each hand-written model and obtain the same predictions and initial parameters as the notebook.

## Scope

Extract Vanilla RNN, LSTM, custom GRU, hidden-state tracking, weight initialization, retention-bias modes, and model construction. Keep initialization in the model area; do not create a registry or extra abstraction hierarchy.

## Dependencies

02. It can proceed alongside 03.

## Files or components affected

Models module, initialization and equation tests, baseline model fixture comparisons.

## Acceptance criteria

- [x] Sequence-first input produces a scalar per batch item; Vanilla RNN, LSTM, and GRU equations match independent calculations; custom GRU uses `(1-z)` for old state and `z` for candidate; `W_z` is the Update Gate linear layer and only `W_z.bias` receives the GRU retention-bias assignment; `W_r` is the Reset Gate layer.

## Required tests

Independent equation cases; zero initial states; output shape; exact seeded initial parameter arrays; orthogonal/Xavier scope over all two-dimensional weights; unchanged biases where specified; zero/one/chrono/numeric gate signs, including LSTM chrono input-gate pairing.

## Behavior-preservation checks

Construct and initialize on CPU before device transfer; preserve constructor RNG order, manual gate equations, custom GRU sign, and tracked hidden tensors. Detached LSTM cell snapshots must not detach the live recurrent cell state.

## Definition of Done

- [x] Every model builds and predicts in isolation, all equation/initialization tests pass, and notebook hash is unchanged.

## Answer

Extracted the three handwritten models and their initialization into [models.py](../../../src/gradient_vanishing/models.py). The factory constructs and initializes on CPU before moving to the requested device. Independent two-step equations and all directly executed model and gate fixtures pass in [test_models.py](../../../tests/test_models.py). The complete test suite passes (59 tests). The original notebook and immutable fixture SHA-256 hashes remain unchanged. The code review found two gaps, both addressed: independent equations now exercise recurrence, and the factory uses direct class selection without a registry.
