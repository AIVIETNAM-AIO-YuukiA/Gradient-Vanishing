# Follow-up 02: Restore evaluation mode correctly

Status: **after behavior-preserving extraction**. Baseline `evaluate` always calls `model.train()` after measuring MSE. A caller that passed an evaluation-mode model therefore receives a training-mode model. The current recurrent classes have no mode-dependent dropout or batch normalization, so the saved metrics are not known to be affected.

After baseline regression tests pass, change evaluation to restore the model's incoming `training` flag, including when evaluation raises. Preserve `torch.no_grad()` and numerical MSE behavior. Keep this change separate from extraction so any numerical difference can be attributed.

Regression tests: call evaluation with a train-mode and an eval-mode model and assert each mode is restored; add an exception-path case and a seeded prediction/MSE comparison. Run the main-grid smoke test to confirm no unintended change.
