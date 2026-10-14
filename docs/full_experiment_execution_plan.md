# Ticket 14: full experiment execution plan (preflight only)

**Status:** awaiting explicit resource and execution approval. No full study,
300-update training run, or full diagnostic sweep has been launched. This plan
audits notebook cells 5, 15, 17, 19, 21, 23, 25, 27, and 28 against the
approved specification and the current CLI. It is an execution plan, not an
empirical reproduction report.

## Source and installation preflight

| Check | Observation on 14 October 2026 |
| --- | --- |
| Branch / HEAD | `refactor/notebook-to-project` / `1cfbb4122e8f53e40612029d176a0eedfb80fb4c` before this plan commit |
| Predecessor | Ticket 13 and its tracker entry are `resolved`; its 136-test wheel validation passed |
| Working tree | No tracked changes at preflight; existing untracked user files, including the Ticket 14 issue, are preserved |
| Notebook SHA-256 | `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44` |
| Immutable fixture SHA-256 | `edc601e85522254471424940869598836cfea940a8d8711580bf8f0eca0818b9` |
| Default JSON SHA-256 | `8d2213395a56c8b38427fd9c1784710f5f698c616dd461518b15e94e4dfcc178` |
| CPU lock SHA-256 | `0549122198a7bc63b5d8bf57e5cbbeaa680258f953be6d15b2de1d34893f6769` |
| Tested wheel | SHA-256 `65842d4e719dedb135b3f51a2433fd6febea5940f6d0b1c138eb609dc8620928`; embedded commit `e9a40fc94c1c7bf76d85b2e2ecf4c6691475d2f5` |

The installed Ticket 13 wheel predates HEAD by documentation and issue updates
only: `docs/commands.md`, `docs/end_to_end_validation.md`,
`docs/installation.md`, and the Ticket 13 issue. `git diff e9a40fc..HEAD --
src pyproject.toml setup.py` is empty. The wheel's computation code therefore
matches current source, but its manifest commit differs from HEAD. Before an
approved full run, build and install a new non-editable wheel from the then-current
committed HEAD, verify its embedded revision, and preserve its wheel SHA-256.
Do not infer the revision from the run directory or from an unrelated checkout.

## Exact notebook matrix and execution order

The notebook orders the main experiment by sequence length `10, 30, 100`, then
model `Vanilla RNN, LSTM, GRU`. For each of the nine cells, it creates one
new Optuna study, then runs all five default seeds before all five tuned seeds.
The selected trial's parameters are reused for the tuned seeds. The fixed test
pair for a length is requested before that length's studies. Later notebook
sections run initial-gradient diagnostics at `T=100`, gate ablation at
`T=100`, and finally the numeric sweep at `T=100`.

| Stage and conditions | Model training invocations | Scheduled optimizer updates | Gradient-only runs |
| --- | ---: | ---: | ---: |
| Optuna: 3 lengths × 3 models × 20 trials | 180 trials | Up to 27,000 (`180 × 150`); pruning may reduce actual updates | 0 |
| Main: 3 lengths × 3 models × 2 settings × seeds 0–4 | 90 | 27,000 (`90 × 300`) | 0 |
| Initial gradients: 3 models × seeds 0–9, `T=100` | 0 | 0 | 30 |
| Gate ablation: 1 Vanilla reference + LSTM/GRU × zero/one/chrono, seeds 0–4 | 35 | 10,500 (`35 × 300`) | 70 (`7 × 10` initial-gradient seeds) |
| Numeric sweep: LSTM then GRU × 17 biases × seeds 0–4 | 0 | 0 | 170 |
| **Maximum scheduled** | **305** | **64,500** | **270** |

The tuning objective minimizes **final validation MSE**. Each study samples
`lr` log-uniformly from `1e-4` to `5e-2`, `hidden_size` from `[16,32,64]`,
and `init_method` from `[orthogonal,xavier,default]`. It uses TPE seed 42,
`MedianPruner(n_startup_trials=5, n_warmup_steps=50)`, and trial seed
`1000 + trial.number` (thus 1000–1019 in each study). A pruned trial retains
its actual update count; a failed study must not be treated as a selected
configuration. No test set enters selection.

Default final runs use `lr=0.001`, hidden size 32, default weight
initialization, and gate mode `one`. Tuned final runs use the study's best
`lr`, hidden size, and initialization, also with gate mode `one`. Each final
run has batch size 128, 300 updates, validation every 10 updates (30 checks),
fixed validation/test size 2048 with split seeds 10000/20000, final-checkpoint
test MSE, and pre-clipping hidden-gradient captures at zero-based updates
`[0,50,150,299]`. There are 90 final test evaluations and 360 captured
training-gradient profiles. Tuning has at most 2,700 scheduled validation
checks; final training has 2,700. The notebook's evaluation-mode side effect
and custom GRU equation remain unchanged.

The ablation trains 35 default-hyperparameter models for 300 updates each:
Vanilla RNN `default`, then LSTM and GRU in `zero`, `one`, `chrono` order.
Validation checks every 10 updates and at the final update; first validation
MSE strictly below `0.01` is recorded, or missing if never reached. It has
1,050 scheduled validation checks and 35 final test evaluations. After each
condition's five training seeds, ten float64 initial-gradient seeds are
measured. `chrono` preserves its random draw and gate-bias assignment order.

The sweep uses exactly `[-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4]`
in ascending order at `T=100`, five seeds per `(model,bias)`. LSTM assigns
`W_f.bias=b`; the custom GRU assigns `W_z.bias=-b` and does not alter `W_r`.
It measures initial hidden-gradient decay only; it has **no per-bias training,
validation MSE, or test MSE**.

The initial-gradient quantity is the batch mean of the hidden-unit L2 norm
of `∂L/∂h_t`, for `t=0..T-1`, with final-prediction mean MSE as loss and full
BPTT. Float64 initial curves divide each timestep norm by the final one;
their geometric mean is plotted. Per-seed decay is `exp(-slope)` from a line
fit to log normalized norms at `t=T//2..T-2`; condition means and sample
standard deviations use the five or ten individual seed values (`ddof=1`).
The float32 main heatmap instead uses tuned seed zero and
`log10(clip(g_t/(g_last+1e-30),1e-10,None))`. These are distinct measured
quantities, not parameter gradients or proof of universal vanishing.

## Expected results and report products

Each stage gets its own unique run directory, manifest, and hashed JSON
results. `main` writes `results/main.json` plus nine
`results/study_{model}_T{length}.json` files. `gradients` should write only
three `results/initial_gradient_{model}_T100.json` files. `ablation` writes
`results/gate_ablation.json`; `sweep` writes
`results/retention_bias_sweep.json`. The final report should contain
`main_summary.csv` (nine model/length rows), `gate_ablation.csv` (seven rows),
`retention_bias_values.csv` (34 model/bias rows), and the notebook-equivalent
mean validation learning curves, T100 initial-gradient curves, T100 tuned
seed-zero training heatmap, and T100 sweep, with PNG and SVG variants and a
hashed `report_index.json`. The reporting module also offers an Optuna trial
table. The CLI accepts one `study` source per report, so generate one combined
report with the first study and eight additional study-only report runs to
cover all nine studies; retain all nine indexes. No report command may train,
tune, or recompute gradients.

## Hardware and feasibility

| Resource | Measured available environment | Consequence |
| --- | --- | --- |
| CPU | 12th Gen Intel Core i5-12450HX, 12 logical processors; PyTorch default 8 intra/inter-op threads | CPU is the validated reference route; run one stage process at a time |
| Memory | 15.71 GiB total, 5.16 GiB available at check; a 10-update T100 hidden-64 calibration process peaked at about 392 MiB working set | Budget at least 4 GiB available before each stage; full-stage cumulative peak is not measured |
| GPU | NVIDIA GeForce RTX 2050, 4,096 MiB VRAM (3,201 MiB free), driver 616.92 | The pinned Torch `2.11.0+cpu` reports CUDA unavailable and `torch.version.cuda=None`; GPU runtime and numerical equivalence are **NOT VERIFIED** |
| Disk | D: 108.84 GiB free of 244.14 GiB total | Reserve 1 GiB for results and manifests, plus package/environment space; no disk constraint is apparent |
| Software | Windows 11, CPython 3.13.0, Torch 2.11.0+cpu, Optuna 4.8.0, NumPy 2.4.0, Matplotlib 3.10.8; `pip check` clean | Matches the pinned CPU validation environment; no CUDA package swap is proposed |

The machine has a GPU, but this environment cannot use it. A CUDA run would
require a separately pinned and validated environment, and cross-device
bitwise identity is not required by the approved specification. **Recommend
the CPU path** for this reproduction. If free RAM drops below 4 GiB or D:
free space below 1 GiB, defer execution rather than run under memory/disk
pressure.

## Timing evidence and estimate

The Ticket 13 corrected-wheel manifests measured, on reduced settings,
approximately 3.9–8.2 seconds for `main`, 3.5–11.8 for `gradients`,
4.1–10.8 for `ablation`, 3.8–5.2 for `sweep`, and 2.3–2.5 for `report`.
Those runs used batch 8, two or three updates, and T10, so they are **not**
linear full-matrix timings.

For this preflight, representative non-destructive CPU calls used the original
batch 128 and fixed validation size 2048, but only ten updates and one
validation check per `(T,model)`; an extra T100 pass sampled hidden size 64.
Observed warm 10-update times in seconds were:

| T / hidden size | Vanilla RNN | LSTM | GRU |
| --- | ---: | ---: | ---: |
| 10 / 32 | first-call 1.41 (includes Torch optimizer startup); warm equivalent not directly measured | 0.073 | 0.067 |
| 30 / 32 | 0.074 | 0.226 | 0.168 |
| 100 / 32 | 0.159 | 0.671 | 0.459 |
| 100 / 64 | 0.258 (another process's first call was 1.57) | 0.622–0.875 | 0.536–0.648 |

At T100, warm float64 initial-gradient probes with two seeds took roughly
0.06 seconds for Vanilla RNN, 0.11–0.12 for LSTM, and 0.09–0.11 for GRU;
the first probe in a process took about 1.63 seconds including startup.
One attempted one-seed diagnostic calibration correctly failed validation
because sample standard deviation needs two seeds; it produced no result
record and was rerun with two seeds. No full experiment condition was run.

Extrapolating the 10-update blocks to the maximum update schedule gives an
optimistic **about 30 minutes of model computation**: roughly 11 minutes
for tuning, 11 for main final runs, 9 for ablation, plus gradient-only work.
Trial-specific hidden sizes, pruning, repeated process startup, file writes,
temperature throttling, and full-run memory behavior are not captured by
that multiplication. Plan **45–90 minutes wall clock**, plus up to 15 minutes
for installation, integrity checks, and report generation; reserve a
**two-hour uninterrupted CPU window**. This is a planning range, not a
measured full-run duration. Pruning may shorten tuning; failures or retries
may extend it. The 10 reduced Ticket 13 run directories together used under
2 MiB of files. Longer histories, 270 T100 curves, and nine studies should
still fit well within a conservative **1 GiB result-artifact reserve**;
actual full storage is NOT VERIFIED.

## Proposed execution after approval

Use a new output root, for example `D:\ILPT\Project\Gradient-Vanishing\artifacts\ticket14_full_<approved-id>`.
Do not place it inside the Ticket 13 reduced-run root. No command below has
been launched as a full stage. Before running, record HEAD, hashes of the
notebook, fixture, default JSON, CPU lock, and newly built wheel. Rebuild the
wheel from committed HEAD, install it into a fresh Python 3.13 environment
with `requirements-cpu.lock`, run `pip check`, CLI help, and the lightweight
regression suite. Verify the wheel's `_build_info.py` commit equals HEAD.

Then, from outside the checkout, set `$cli` to the installed
`gradient-vanishing.exe` and `$root` to the approved output root:

```powershell
& $cli main --output-dir $root --allow-full
& $cli gradients --set 'sequence_lengths=[100]' --output-dir $root --allow-full
& $cli ablation --output-dir $root --allow-full
& $cli sweep --output-dir $root --allow-full
```

These are the real Ticket 12 CLI arguments. The `gradients` override is
**required**: the CLI's package default would measure all three lengths,
whereas notebook cell 23 measures only `max(SEQ_LENS)=100`. This override
records the notebook-equivalent diagnostic condition in its effective config,
but current CLI classification will label that stage `reduced_or_modified`;
that label is a metadata limitation, **not evidence of reduced gradient
measurement**. Before reporting full reproduction, verify the manifest's
actual conditions and explicitly annotate this limitation. Do not silently
run the extra T10/T30 curves or relabel the manifest as full.

After each stage, select its **specific** completed directory from the CLI
output. Check manifest `status=complete`, resolved CPU device, source commit,
default budgets (or the documented T100 diagnostic override), model/seed/order
counts, all artifact SHA-256 values, and result structure before starting
the next stage. Do not select a run merely because it is the newest. Example
report syntax, with the exact completed paths substituted manually:

```powershell
& $cli report --output-dir $root `
  --source "main=$mainJson" `
  --source "ablation=$ablationJson" `
  --source "sweep=$sweepJson" `
  --source "study=$firstStudyJson" `
  --source "initial_gradients=$vanillaGradientJson" `
  --source "initial_gradients=$lstmGradientJson" `
  --source "initial_gradients=$gruGradientJson"
```

For each of the other eight saved study files, run
`& $cli report --output-dir $root --source "study=$studyJson"` in a separate
unique report run. Source file paths and hashes in each `report_index.json`
must match selected stage artifacts. Retain the raw JSON, manifest, figure,
table, report-index, wheel, config, and lock hashes together in the eventual
reproducibility report.

Interruption is **stage-granular**. The CLI creates unique directories and
atomic JSON writes; it does not implement a mathematical checkpoint or resume
inside a study, a main grid, an ablation, or a sweep. A caught failure may
save a partial record, while a killed process may leave a `running` manifest
with no final raw result. Preserve such directories as evidence. On restart,
rerun the **entire failed stage** in a new directory under the same approved
root; do not merge partial trial/run records or repeat only selected seeds.
Record the abandoned run ID and reason, and select exactly one completed
stage run for the final reports. This avoids duplicate conditions in reported
aggregates and does not alter notebook RNG or training semantics.

## Acceptance and interpretation checks

1. Re-run the lightweight fixture and direct-notebook tests before full
   execution. Confirm the original source and fixture hashes before and after.
2. Verify nine studies, 20 intended trials each, sampler/pruner settings,
   trial seeds, trial states (complete/pruned/failed), and best completed trial
   selection. Preserve partial/failed studies; do not invent best parameters.
3. Verify 90 main final runs, five seeds per setting, original length/model
   and default-then-tuned order, effective parameters, 300 updates, 30
   validation checks, four gradient captures, test split isolation, and final
   model mode behavior. Recompute mean and `ddof=1` spread from raw per-seed
   test values.
4. Verify 35 ablation training runs and 70 ablation gradient-only runs, gate
   bias vectors/signs, threshold timing, raw test values, decay fits, and
   `ddof=1` spreads. Verify 170 sweep measurements, bias order, per-seed
   curves and decay means; do not seek nonexistent per-bias test MSE.
5. Verify three T100 initial-gradient records with ten seeds each, forward
   timestep ordering, float64 precision, normalization, fit window, and
   separate float32 trained-gradient heatmap semantics.
6. Regenerate reports only from selected saved results. Compare table entries
   and representative plotted points directly to raw JSON; check 9 main rows,
   7 ablation rows, 34 sweep rows, all required PNG/SVG files, nine study
   tables, report source hashes, and explicit missing-condition reporting.
7. Compare selected full-run observations with direct original-notebook
   execution where practical and with the immutable fixture at its recorded
   small cases. Compare full aggregates with the notebook's saved outputs as
   **historical, unverified observations**, not as an oracle. Classify any
   discrepancy as implementation, environment, stochastic variation, or
   possibly stale saved notebook output. Same-implementation repeatability
   alone does not establish independent notebook equivalence.

Use the approved float32 tolerance `rtol=1e-5, atol=1e-6` and float64
`rtol=1e-8, atol=1e-10` for comparable computed values in the pinned CPU
environment; compare counts, labels, seeds, state, and hashes exactly. Do not
claim cross-device bitwise equality, a universal gradient-vanishing law, or
that tuned models universally outperform defaults. Document every failed,
pruned, and unrun condition in the later Ticket 14 report.

## Approval recommendation

Approve a **single-process, pinned-CPU, two-hour window**, with at least
4 GiB free RAM and 1 GiB free disk at start, for the four stages and saved-
result reports above. Rebuild the wheel from the approved committed HEAD
before execution. Approve the explicit T100-only diagnostic override and its
metadata caveat. If a fully automatic `full_configuration` label for that
diagnostic stage is required, resolve that metadata issue in a separately
reviewed change before launching the matrix. Until explicit approval, the
full empirical reproduction and GPU path remain **NOT VERIFIED**.
