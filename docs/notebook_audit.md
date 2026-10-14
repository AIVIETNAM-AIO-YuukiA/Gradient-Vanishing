# Notebook audit

Baseline: [`Notebooks/GradientVanishingAnalysis.ipynb`](../Notebooks/GradientVanishingAnalysis.ipynb). Audited as the source of truth, without editing or executing it. SHA-256 at audit time: `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44` (663,298 bytes; 29 cells: 15 code, 14 Markdown). Cell numbers below are **zero-based JSON indices**; the notebook's own comments number code cells 1–14.

## Pipeline summary

The research question is how Vanilla RNN, LSTM, and GRU retain an initial scalar across sequences of length 10, 30, and 100, and how their gradient flow and gate initialization differ. Each input has shape `(T, batch, 1)`: `x[0]` is sampled from `Uniform[-1, 1]`, and all later inputs are zero. The target is that initial scalar. A predictor without memory has theoretical MSE `1/3`.

1. Install Optuna in the notebook environment; import libraries, select CPU/CUDA, request deterministic Torch algorithms, and define seeding.
2. Define constants and generate fresh synthetic training batches. Cache fixed validation and test sets separately for each `T`, using CPU generators seeded `10_000` and `20_000` respectively. The same first-element values recur across sequence lengths because the same seed is reused for each `T`.
3. Build hand-written recurrent models. Each emits a scalar from its final hidden state and retains intermediate hidden tensors for gradient measurement. LSTM also stores detached cell states.
4. Initialize weights and gate biases. The main comparison uses retention-bias mode `one` for both configurations called `default` and `tuned`; `default` means default **hyperparameters**, not default gate bias.
5. For each `(T, architecture)`, run one 20-trial Optuna study over learning rate, hidden size, and weight initialization. A trial trains for up to 150 optimizer updates on one trial seed; its last validation MSE is the objective. Median pruning may shorten trials.
6. Train the default and tuned configurations for 300 optimizer updates, each with five seeds (`0`–`4`), evaluate the final model on fixed test data, and retain loss and selected gradient histories in memory.
7. Display a test-MSE summary and validation learning curves; compare initial gradient ratios over ten seeds in float64; plot a during-training gradient heatmap for the tuned `T=100` runs at seed 0.
8. At `T=100`, compare Vanilla RNN with LSTM/GRU retention-bias modes `zero`, `one`, and `chrono` using default hyperparameters; report test MSE, first checked validation step below `0.01`, and initial gradient decay. Finally, sweep numeric retention biases from `-4` to `4` against measured decay.

There is no data ingestion, model checkpoint, exported result, or standalone inference API. Inference is `model(X)`; `evaluate` computes MSE with `torch.no_grad()`.

### Observed saved outputs

The saved run reports CPU execution and Optuna 5.0.0 installation. At `T=100`, tuned LSTM reports test MSE `0.0002 ± 0.0003`; tuned Vanilla RNN and GRU report about `0.341`. In the `T=100` ablation, chrono LSTM and GRU report `0.0009 ± 0.0006` and `0.0003 ± 0.0002`, versus about `0.341` for zero/one. These are **saved outputs**, not results reproduced by this audit. All code-cell `execution_count` fields are null, so metadata alone cannot establish execution chronology. The finite fixed test set's near-zero-predictor MSE is about `0.341`, while `1/3` is the population baseline.

## Cell-to-component mapping

| Cell | Notebook content | State or dependency created | Used by |
| --- | --- | --- | --- |
| 0 | `!pip install optuna` | Runtime package installation | 3, 15 |
| 1–2 | Problem, baseline, research questions, setup heading | Experimental intent | All analyses |
| 3 | Imports, `device`, determinism, `set_seed` | Global libraries/device/RNG policy | 7 onward |
| 4–5 | Experiment constants | `SEQ_LENS`, seeds, budgets, `DEFAULT_CONFIG`, `MAIN_GATE_INIT` | 7 onward |
| 6–7 | Synthetic data | `generate_memorization_task`, `make_fixed_set`, cached `get_eval_sets`, `MEMORYLESS_BASELINE` | 13, 17, 21, 23, 27 |
| 8–9 | Model equations and implementations | `TrackerBase`; `VanillaRNN_Tracker`, `LSTM_Tracker`, `GRU_Tracker`; `MODEL_CLASSES` | 11 onward |
| 10–11 | Initialization | `apply_weight_init`, `apply_gate_init`, `build_model` | 13, 23 |
| 12–13 | Training and evaluation | `evaluate`, `train_model` | 15, 17, 27 |
| 14–15 | Hyperparameter search | `tune_model`, Optuna objective | 17 |
| 16–17 | Main experiment | `results[T][model][setting]`, `best_params`, `RECORD_GRAD_EPOCHS` | 19, 21, 25, 27 |
| 18–19 | Summary | `mean_std`, `summary` DataFrame | 27 also uses `mean_std` |
| 20–21 | Learning curves | `colors`, `SETTINGS`, plot | 23 and 28 also use `colors` |
| 22–23 | Initial gradient analysis | `measure_init_gradients`, `decay_factor_from_curves`, `T_GRAD` | 25, 27, 28 |
| 24–25 | Training gradient heatmap | Plot from `results`, `T_GRAD`, `RECORD_GRAD_EPOCHS` | Presentation only |
| 26–27 | Gate ablation | `ABLATION_T`, `GATE_INITS`, `THRESHOLD`, `epochs_to_threshold`, `fmt_speed`, `ablation` | Presentation only |
| 28 | Numeric bias sweep | `T_SWEEP`, sigmoid reference, plot | Presentation only |

Run the code cells in order in a fresh kernel. Later cells rely on globals rather than explicit inputs: cell 17 needs 3, 5, 7, 9, 11, 13, 15; cell 19 needs 17; cell 21 needs 17; cell 23 needs 21 for `colors`; cell 25 needs 17, 21, and 23; cell 27 needs 17, 19, and 23; cell 28 needs 21 and 23. Re-running cell 17 replaces `results` and `best_params`; re-running cell 7 resets the validation/test cache. Executing cells out of order can silently combine state from different configurations.

## Dependency inventory and environment assumptions

| Dependency | Purpose | Baseline evidence |
| --- | --- | --- |
| Python and Jupyter/IPython | Notebook execution, `!pip`, `display` | `python3` kernelspec; Colab metadata; saved pip output references Python 3.13 paths |
| PyTorch | Tensor generation, autodiff, layers, Adam, gradient clipping, CUDA | Imported as `torch`, `nn`, `F`, `optim`; version unpinned |
| NumPy | Statistics, curves, bias sweep | Imported; saved pip output says NumPy 2.1.3 in that environment |
| pandas | Summary and ablation tables | Imported; version unpinned |
| Matplotlib, seaborn | Learning-curve, gradient, heatmap plots | Imported; versions unpinned |
| Optuna | TPE search and median pruning | Installed by cell 0; saved output says 5.0.0 |
| Python `os`, `random` | CuBLAS setting and seeding | Standard library |

No dataset path or file-system path is hard-coded in source cells. There is no requirements file, lockfile, Python version declaration, installation command for Torch/plotting libraries, or saved artifact location. Cell 0 requires package-network access and notebook shell syntax. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before importing Torch; CUDA availability determines device. Training data is generated on CPU and moved to the selected device. Reproducibility across Torch/library versions, CPU/GPU, and hardware is unverified. The full notebook is compute-heavy: up to 27,000 tuning updates before pruning, 27,000 main-run updates, 10,500 ablation updates, and 170 initial-gradient backward passes, in addition to plotting.

## Training, evaluation, inference, and metrics

- `train_model` reseeds Python, NumPy, and Torch for each run; initializes a new model; uses Adam, fresh batch of 128 each update, MSE loss, and global parameter-gradient clipping at norm 5. Validation uses 2,048 fixed examples every 10 updates (plus the final update). Its `epoch` variable counts optimizer updates, not full passes through a dataset.
- Tuning search space: `lr` log-uniform from `1e-4` to `5e-2`; `hidden_size` in `{16,32,64}`; `init_method` in `{orthogonal,xavier,default}`. One study per model and `T`, 20 trials, TPE sampler seed 42, median pruner with five startup trials and 50 warm-up steps. Gate bias is excluded from tuning.
- The final test set has 2,048 fixed examples per `T`. `test_mse` is measured only after the final update of each five-seed run; tables report arithmetic mean and **sample** standard deviation (`ddof=1`). Learning curves show mean validation MSE but do not plot the stated standard deviation. Threshold speed is the first validation check below `0.01`, so its resolution is 10 updates.
- Gradient metric: mean across batch of `L2` norm of `dL/dh_t`; initial-gradient plots divide by the last-step value, use float64 and geometric mean over ten seeds, then fit `r` from log gradients over `t = T//2 ... T-2`. Training heatmap uses float32 gradients recorded before clipping/optimizer step, normalizes to the last step, then clips display to `10^-10`. LSTM cell-state gradients are not measured; hidden-state gradients alone do not describe every gradient path through the LSTM.
- Retention-bias semantics: `W_z` is an `nn.Linear` layer with **both** `weight` and `bias` parameters. Its output defines the update gate `z = sigmoid(W_z([x_t,h_{t-1}]))`. The ablation writes **`model.W_z.bias`**, not the layer's weights and not `W_r`. `W_r` defines the Reset Gate. Because this custom GRU updates `h_t = (1-z)h_{t-1} + z h_tilde`, positive abstract retention bias `b` is implemented as `W_z.bias = -b`. LSTM uses `W_f.bias = b`; chrono also uses `W_i.bias = -b`. Preserve these equations and signs in a refactor.

## Refactor risks and existing issues

### Behaviors to preserve in the first refactor

Keep sequence-first tensor shapes, synthetic distribution and fixed-set seeds, cache lifetime within one run, manual recurrent equations and gate sign convention, initialization order (construct on CPU, initialize, move to device), five-seed run grid, 150/300-update budgets, validation/test separation, final-checkpoint evaluation, sample standard deviation, and the stated gradient measurement definitions. Preserve the `default` hyperparameter setting paired with gate mode `one`. These are baseline semantics even where alternative designs might improve research quality.

### Bugs and documentation errors for separate follow-up work

1. **GRU terminology mismatch.** Intro and later prose call the intervention “Reset Gate,” but code changes `W_z.bias` on the Update Gate. `W_r` remains at its ordinary initialization. See [terminology follow-up](followups/01-gru-gate-terminology.md).
2. **`evaluate` changes the model's mode.** It calls `model.eval()` and then unconditionally `model.train()`. An input model already in evaluation mode is left in training mode. Current models have no dropout or batch normalization, so saved MSE results are not known to change. See [mode-restoration follow-up](followups/02-evaluate-mode.md).
3. **Learning-curve description overstates the chart.** Source comment says “mean ± std” over seeds, but cell 21 plots only the mean. See [uncertainty-display follow-up](followups/03-learning-curve-uncertainty.md).

### Methodological and operational risks, not silently reclassified as bugs

- A single 150-update, single-seed trial selects a configuration used for 300-update, five-seed reports. Saved results show tuned Vanilla RNN at `T=10` (`0.1163 ± 0.1391`) and tuned LSTM at `T=30` (`0.0686 ± 0.1523`) perform worse than their default settings despite near-zero trial validation scores. Preserve selection behavior initially; investigate robustness separately.
- The baseline `1/3` is a population expectation; the fixed test sample differs. Comparing `0.3409` directly to `0.3333` without the sampled zero-predictor MSE can mislead.
- Saved outputs cannot prove a clean, in-order run. No package lock, results manifest, checkpoint, or automated tests exist. The notebook's narrative and figures may reflect a particular unrecorded environment.
- A parameter refactor can change Torch RNG consumption and initial values even with unchanged formulas. Validate both structural equations and seeded small-case numerical outputs. A different GRU convention, input shape, epoch meaning, or best-checkpoint selection would change the experiment.

## Reproducibility checklist

- [ ] Record baseline notebook hash and inspect `git diff` to ensure it remains untouched.
- [ ] Pin a supported Python version and exact Torch, NumPy, pandas, Matplotlib, seaborn, Optuna, and notebook-runner versions; record CPU/GPU and CUDA versions.
- [ ] Create a fresh environment and run from a fresh kernel in notebook order; record installation and execution commands.
- [ ] Set `CUBLAS_WORKSPACE_CONFIG` before importing Torch, set deterministic algorithms, and record any backend errors or unsupported operations.
- [ ] Record all constants, device, data type, seed list, sampler/pruner settings, and selected `best_params` for each `(T, model)`.
- [ ] Check synthetic tensor shapes/range/zero tail and fixed validation/test repeatability and separation.
- [ ] Check that five-seed default/tuned results use the same fixed test set for each `T`, and that test data is not used by the tuning objective.
- [ ] Capture per-run train/validation losses, final test MSE, initialization gradient curves, heatmap inputs, and ablation rows with a run manifest.
- [ ] Compare the refactor with the baseline on deterministic small cases before the full experiment; permit documented floating-point tolerance across platforms.
- [ ] Re-run the full grid only after its runtime and hardware budget is agreed; compare scientific conclusions and numerical reports without demanding bitwise identity across devices.
