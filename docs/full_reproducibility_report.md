# Ticket 14: full CPU experiment matrix and reproducibility report

**Execution:** 14 October 2026, 14:52:48–15:17:54 UTC. **Outcome:** all notebook-equivalent conditions completed. This is full matrix **execution coverage** in the pinned CPU environment; independent numerical equivalence to every historical notebook output is not established. The notebook's saved output is treated as an unverified historical observation.

## Provenance and environment

The immutable source was `Notebooks/GradientVanishingAnalysis.ipynb`, SHA-256 `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44`. The baseline fixture `tests/fixtures/notebook_baseline.json` remained SHA-256 `edc601e85522254471424940869598836cfea940a8d8711580bf8f0eca0818b9`. Execution source commit was `8298ac766cf8af9a8316bb1d67d8174ed25546d3`; the newly built, non-editable wheel embedded this commit and had SHA-256 `02eab7276c256de91b4aa0a15c4b9928d13afcd5e6ff82f751880414008f020f`. The pinned CPU lock had SHA-256 `0549122198a7bc63b5d8bf57e5cbbeaa680258f953be6d15b2de1d34893f6769`; `pip check` passed. All four stage manifests record the same source commit and notebook reference hash.

The machine was Windows 11 with a 12th Gen Intel Core i5-12450HX (12 logical processors), 15.71 GiB system RAM, CPython 3.13.0, PyTorch 2.11.0+cpu, Optuna 4.8.0, NumPy 2.4.0, and Matplotlib 3.10.8. Torch reported `cuda.is_available() == False`; **GPU execution is not verified**. Available RAM was at least 5.26 GiB before the first stage and 5.88–5.97 GiB at subsequent boundaries; free disk remained above 107.88 GiB. These exceeded the approved 4 GiB and 1 GiB gates. The four stages ran sequentially, with no duplicate full-matrix process active at launch. The fresh wheel was installed in `.tmp/ticket14_full_env`; its CLI was invoked from outside the repository checkout.

The artifact root is `.tmp/ticket14_run/full_matrix_20261014T215236/`, separate from all Ticket 13 reduced outputs. It is ignored by Git and must be retained or archived separately when sharing this report. The machine-readable [matrix coverage manifest](../.tmp/ticket14_run/full_matrix_20261014T215236/matrix_coverage.json), SHA-256 `6a11289d3d5265ff5de6582922ae5bde57eac8dac84ed951961b9872247edac5`, explicitly records completed, pruned, failed, and unrun counts and links the four stage manifests. The complete [artifact inventory](../.tmp/ticket14_run/full_matrix_20261014T215236/artifact_inventory.json) lists 63 files with byte sizes and SHA-256 hashes; its own SHA-256 is `7d0575cf458046a939316814a17aecb8250e7ef4dcc668fd2f111a0178761c0b`. Total inventoried size is 7,661,667 bytes (7.31 MiB). Every stage and report manifest artifact hash was independently recalculated after completion.

## Matrix coverage and timing

| Stage | Approved conditions | Completed | Actual stage time |
| --- | --- | --- | ---: |
| Optuna, inside main | 3 lengths × 3 models × 20 trials | 180: 105 complete, 75 pruned, 0 failed, 0 unrun | included below |
| Main final training | 3 lengths × 3 models × default/tuned × 5 seeds | 90 of 90; 300 updates each | 835.539 s for tuning and main |
| Initial hidden gradients | T=100 × 3 models × 10 seeds | 30 of 30 | 4.566 s |
| Gate ablation training | T=100 × 7 conditions × 5 seeds | 35 of 35; 300 updates each | 421.917 s |
| Gate ablation initial gradients | 7 conditions × 10 seeds | 70 of 70 | included above |
| Numeric bias sweep | T=100 × LSTM/GRU × 17 biases × 5 seeds | 170 of 170 | 11.854 s |
| Saved-result reporting | one combined report and eight further study reports | nine of nine | 8.2 s manifest time |

The approved **270 gradient-only runs** comprise 30 initial, 70 ablation, and 170 sweep measurements. All 90 main runs also produced four training-gradient captures at zero-based updates 0, 50, 150, and 299 (360 profiles). Complete Optuna trials used 150 updates; pruned trials stopped at their recorded validation steps. The nine studies consumed **20,100 actual tuning updates**, versus 27,000 maximum scheduled. Main and ablation consumed 27,000 and 10,500 updates, respectively: **57,600 actual optimizer updates** overall. The first experiment start to final report completion took **1,506.327 seconds (25 minutes 6 seconds)**, including stage verification gaps; the four computational stages themselves took 1,273.875 seconds. Installation, final regression tests, and independent notebook checks were separate from that interval.

Each stage result has `status=complete`. Optuna trial records retain individual `complete` and `pruned` states, parameters, objective values, and intermediate values. There were no failed or unrun conditions. No stage was resumed or merged from a partial run. The validation and test fixed sets used seeds 10000 and 20000, respectively; tuning objective and pruning used final validation MSE and never test MSE. Final training used batch 128, 300 updates, validation every ten updates, Adam and the original evaluation-mode side effect. Per-condition means and **sample** standard deviations were recomputed from all five raw test MSEs or decay values (`ddof=1`).

The `gradients` command used the approved `sequence_lengths=[100]` override because the notebook's initial-gradient cell measures only T=100. The CLI's generic scope classifier marks that manifest `reduced_or_modified`; the recorded 3 × 10 condition set is the **full notebook-equivalent initial-gradient condition**, not a scientifically reduced run. The other three experiment manifests say `full_configuration`. We retained the raw scope field and document its meaning here rather than rewriting provenance.

## Scientific results

Main test MSE values below are five-seed means ± sample standard deviations. The tuned settings reuse each study's selected completed trial. Learning curves, raw per-seed values, selected parameters, and T=100 trained hidden-gradient heatmaps are preserved in the JSON and figures.

| T | Model | Default test MSE | Tuned test MSE |
| ---: | --- | ---: | ---: |
| 10 | Vanilla RNN | 0.0003 ± 0.0003 | 0.0006 ± 0.0010 |
| 10 | LSTM | 0.0008 ± 0.0001 | 0.0006 ± 0.0008 |
| 10 | GRU | 0.0001 ± 0.0000 | 0.0000 ± 0.0000 |
| 30 | Vanilla RNN | 0.3409 ± 0.0004 | 0.1381 ± 0.1879 |
| 30 | LSTM | 0.0002 ± 0.0001 | 0.1003 ± 0.1516 |
| 30 | GRU | 0.0011 ± 0.0001 | 0.0005 ± 0.0011 |
| 100 | Vanilla RNN | 0.3409 ± 0.0004 | 0.2356 ± 0.1413 |
| 100 | LSTM | 0.3409 ± 0.0002 | 0.0682 ± 0.1503 |
| 100 | GRU | 0.3409 ± 0.0002 | 0.1365 ± 0.1868 |

At T=100, the seven gate conditions in notebook order were Vanilla default; LSTM zero, one, chrono; GRU zero, one, chrono. Their test MSE means were respectively 0.3409, 0.3409, 0.3409, 0.0009, 0.3409, 0.3409, and 0.0003. The count of seeds reaching validation MSE strictly below 0.01 was respectively 0, 0, 0, 5, 0, 0, and 5; mean first successful update was 238 for LSTM chrono and 252 for GRU chrono. Initial-gradient decay means were 0.467, 0.629, 0.853, 0.961, 0.635, 0.800, and 0.982, respectively. Exact values and uncertainty are in `gate_ablation.csv` and the raw result.

The numeric sweep measured **initial hidden-gradient decay**, not trained performance, at all 17 values `-4, -3.5, ..., 4`. LSTM assigns `W_f.bias=b`; the custom GRU assigns `W_z.bias=-b` in its `h_t=(1-z_t)h_prev+z_t h_tilde` blend. The fitted per-seed decay is `exp(-slope)` for a least-squares line over log normalized hidden-gradient norms at `t=T//2..T-2`; table rows are five-seed means and sample standard deviations. At b=-4, 0, +4, LSTM mean decay was 0.152, 0.624, 0.996, and GRU mean decay was 0.292, 0.631, 0.982. The 34-row table and raw curves contain every intermediate bias. A decay factor describes this measured initial gradient profile only. It is neither test accuracy nor evidence that a bias universally improves learning.

## Saved outputs and hashes

These paths are relative to the artifact root. Manifests in each run directory provide complete artifact SHA-256 lists, effective configuration, seed/model/length lists, timing, source commit, environment, status, and errors (none here). The inventory includes all nine study JSON files and nine report indexes.

| Run directory | Primary result or index | SHA-256 |
| --- | --- | --- |
| `main-20261014T145248-1791989568605551800-8548/` | `results/main.json` | `5488f215e86fc04262b8e36bfbd2c9759b287f17bfc3499221da6a57859636a4` |
| `gradients-20261014T150736-1791990456949266400-21120/` | `results/initial_gradient_vanilla_rnn_T100.json` | `546ae45d67f631b8cedc1bc67a4086cb2f6e3d896fe0e81ed35cd9b6e7b49cbe` |
| same | `results/initial_gradient_lstm_T100.json` | `ff0ec07dc376c68f8c1c46dd6e228138ec64d8bb5de0a1c21f7e7e5b55f9fa52` |
| same | `results/initial_gradient_gru_T100.json` | `55caa9a49b8a3cd9ef85e3a7a32f804756ca5348936b2e2e218ee0c76023ec67` |
| `ablation-20261014T150812-1791990492091660900-24940/` | `results/gate_ablation.json` | `99d6fb96d443e67b898f05d230f02a88523970e86170bc8e3f3f2aacca5b7a3b` |
| `sweep-20261014T151649-1791991009062308200-9624/` | `results/retention_bias_sweep.json` | `15fbc17bae728d4e930155c4a6aea6deb6c05a0c68e019f5f5d9a543b438bc` |
| `report-20261014T151737-1791991057329882400-5028/` | `reports/report_index.json` | `4bd15689245586d81787c5185e6ce615682def2cb561b89c89787a5275d3933d` |

The combined report directory contains `main_summary.csv` (nine rows), `gate_ablation.csv` (seven), `retention_bias_values.csv` (34), one `optuna_trials.csv` (20 trials), and these four figures, each as PNG and SVG: `main_learning_curves`, `initial_gradient_curves_T100`, `training_gradient_heatmap_T100`, and `retention_bias_sweep_T100`. Eight additional study-only report directories contain the other 160 Optuna trial rows; across nine reports all 180 are represented. All 13 combined-report artifacts and all study report artifact/source hashes matched. Selected CSV values and plotted points matched raw saved JSON. Reporting ran only on saved sources and did not invoke training, tuning, or gradient computation. The generated figures were visually checked for legibility and correct axes.

## Equivalence evidence and discrepancies

**Immutable fixture:** The final 136-test suite includes the stored, original-notebook baseline regressions for data, models, seeded training, and gradients. It passed in the wheel's pinned CPU environment (`136 passed in 68.65s`, zero failures). The source notebook and fixture hashes remained unchanged.

**Direct original notebook:** Executing the untouched notebook definitions for one **full** Vanilla RNN T=10 Optuna study (20 trials, 150-update budget) matched the corresponding modular study exactly: every suggestion, `complete`/`pruned` state, intermediate validation objective, final objective, best trial, and best parameters (maximum numerical difference 0). A direct notebook LSTM chrono ablation run at T=100, seed 0, 300 updates also matched the saved modular run exactly: initial gate vectors, all 300 training losses, all 30 validation losses and epochs, test MSE `0.0007525503169745207`, and final training mode (maximum numerical difference 0). The ignored verification records are `.tmp/ticket14_run/direct_full_study_t10_result.json` and `direct_full_ablation_lstm_result.json`. These checks independently support the extracted code for the examined full conditions; they do not independently rerun all 305 training invocations.

**Historical saved notebook output:** The notebook's displayed main table matches all **nine default** mean±standard-deviation cells at four decimals, while **one of nine tuned** displayed cells (GRU T=10) matches; the other eight tuned cells differ. All nine historically displayed selected parameter sets differ from this execution. The notebook's displayed seven ablation rows match the newly generated rows at the displayed precision for test MSE, success timing, and decay. The saved notebook plots do not expose sufficient numeric arrays to independently compare every plotted point. A fresh direct execution of the notebook's full T=10 Vanilla study produced the new modular trial history exactly, while its historical saved best parameters differ. The notebook's saved cell 3 says `Device: cpu`, so a CPU-versus-GPU switch is **not supported** as the explanation for these saved results. Its saved cell 0 installed `optuna-5.0.0`; this run used pinned Optuna 4.8.0. That version difference is a concrete candidate for a changed TPE trial trajectory, but it was not isolated in an otherwise identical environment. For the directly examined condition, the discrepancy is **historical output/environment or stochastic execution context, not a demonstrated implementation difference**. Other historical package versions and the notebook's cell execution state are not fully recoverable, so the precise cause cannot be assigned between version variation and stale saved output. No algorithm or RNG setting was changed to force the displayed numbers to match. The other eight tuning studies have no full direct-notebook trial-history comparison and remain numerically unverified against independent notebook execution.

**Scientific limits:** The complete approved matrix ran on one CPU environment and five final training seeds. It does not establish device-independent numbers, GPU behavior, or universal superiority of tuned settings or gate biases. The sweep records only untrained hidden-gradient decay, with no per-bias validation/test MSE. Initial float64 gradient profiles and captured float32 trained-model gradients have distinct definitions and should not be combined as one metric. The notebook's known evaluation-mode side effect remains part of the reproduction.

## Reproduction procedure

From a clean checkout of source commit `8298ac7`, use the pinned CPU lock and build/install a non-editable wheel as in [usage.md](usage.md). Verify the wheel's embedded commit, `pip check`, notebook/fixture hashes, at least 4 GiB available RAM, at least 1 GiB free disk, and that no duplicate full run is active. These are the exact stage invocations used on this machine; for a new execution, choose a new unique `$root`:

```powershell
$projectRoot = 'D:\ILPT\Project\Gradient-Vanishing'
$cli = Join-Path $projectRoot '.tmp\ticket14_full_env\Scripts\gradient-vanishing.exe'
$root = Join-Path $projectRoot '.tmp\ticket14_run\full_matrix_20261014T215236'
& $cli main --output-dir $root --allow-full
& $cli gradients --set 'sequence_lengths=[100]' --output-dir $root --allow-full
& $cli ablation --output-dir $root --allow-full
& $cli sweep --output-dir $root --allow-full
```

Select the four **specific completed run directories** from their output; verify manifest `status`, effective configuration, artifact hashes, and cardinalities before the next stage. Never merge a partial run into a completed stage or overwrite prior artifacts. The reporting invocation below uses the actual selected run directories and saved results. The first study is included in the combined report; the remaining eight each receive a separate study report:

```powershell
$mainResults = Join-Path $root 'main-20261014T145248-1791989568605551800-8548\results'
$gradientResults = Join-Path $root 'gradients-20261014T150736-1791990456949266400-21120\results'
$ablationResults = Join-Path $root 'ablation-20261014T150812-1791990492091660900-24940\results'
$sweepResults = Join-Path $root 'sweep-20261014T151649-1791991009062308200-9624\results'
& $cli report --output-dir $root `
  --source "main=$(Join-Path $mainResults 'main.json')" `
  --source "ablation=$(Join-Path $ablationResults 'gate_ablation.json')" `
  --source "sweep=$(Join-Path $sweepResults 'retention_bias_sweep.json')" `
  --source "study=$(Join-Path $mainResults 'study_vanilla_rnn_T10.json')" `
  --source "initial_gradients=$(Join-Path $gradientResults 'initial_gradient_vanilla_rnn_T100.json')" `
  --source "initial_gradients=$(Join-Path $gradientResults 'initial_gradient_lstm_T100.json')" `
  --source "initial_gradients=$(Join-Path $gradientResults 'initial_gradient_gru_T100.json')"
$otherStudies = @('study_lstm_T10.json', 'study_gru_T10.json',
  'study_vanilla_rnn_T30.json', 'study_lstm_T30.json', 'study_gru_T30.json',
  'study_vanilla_rnn_T100.json', 'study_lstm_T100.json', 'study_gru_T100.json')
foreach ($study in $otherStudies) {
  & $cli report --output-dir $root --source "study=$(Join-Path $mainResults $study)"
}
```

Reporting requires no compute stage. For a fresh run, choose a new `$root` **and replace all four hardcoded result directory names** with that run's verified stage directories before reporting. Rerunning the commands creates fresh unique directories; compare result hashes and raw values before selecting an authoritative set. The original notebook and baseline fixture must stay untouched.
