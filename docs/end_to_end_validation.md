# Ticket 13 end-to-end validation — pinned CPU environment

Status: **structurally validated; full empirical reproduction NOT VERIFIED**.
This report records actual wheel-installed executions on 14 October 2026.
The unchanged notebook is the mathematical reference; reduced runs do not
establish its full-matrix research conclusions.

## Environment and protected sources

| Item | Observed |
| --- | --- |
| OS / Python | Windows 11 (`Windows-11-10.0.26300-SP0`), CPython 3.13.0 |
| Device / Torch | CPU; PyTorch `2.11.0+cpu`; CUDA unavailable; deterministic algorithms enabled during experiments |
| Other direct versions | Optuna 4.8.0, NumPy 2.4.0, Matplotlib 3.10.8, pandas 2.3.3 |
| Dependency check | All 38 lock-file packages matched their pinned versions; `pip check` found no broken requirements |
| Source notebook SHA-256 | `71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44` |
| Immutable baseline SHA-256 | `edc601e85522254471424940869598836cfea940a8d8711580bf8f0eca0818b9` |
| Wheel source revision | The first validation wheel was built from the Ticket 12 checkout; its manifests record `git_commit: null`. After code review, a rebuilt wheel embedded source commit `e9a40fc94c1c7bf76d85b2e2ecf4c6691475d2f5`. A new outside-checkout gradient run recorded that exact commit. The initial eight experiment runs and two report runs remain computational evidence, but their source-revision provenance is incomplete. |

The fresh environment was `.tmp/ticket13_validation/venv2`; it was created
without access to the previous `.venv` packages. These installation commands
were executed from the checkout (the first sandboxed dependency attempt could
not reach the package index; the authorized retry completed):

```powershell
$env:TEMP=(Resolve-Path .tmp/ticket13_validation/tmp).Path
$env:TMP=$env:TEMP
py -3.13 -m venv .tmp/ticket13_validation/venv2
.tmp\ticket13_validation\venv2\Scripts\python.exe -m pip install -r requirements-cpu.lock
.tmp\ticket13_validation\venv2\Scripts\python.exe -m pip wheel --no-deps --no-build-isolation --wheel-dir .tmp/ticket13_validation/wheels .
.tmp\ticket13_validation\venv2\Scripts\python.exe -m pip install --no-deps .tmp/ticket13_validation/wheels/gradient_vanishing_research-0.1.0-py3-none-any.whl
.tmp\ticket13_validation\venv2\Scripts\python.exe -m pip check
```

The review correction was checked with a second wheel build and non-editable
reinstallation into the same isolated environment:

```powershell
.tmp\ticket13_validation\venv2\Scripts\python.exe -m pip wheel --no-deps --no-build-isolation --wheel-dir .tmp/ticket13_validation/revision_wheels .
.tmp\ticket13_validation\venv2\Scripts\python.exe -m pip install --no-deps --force-reinstall .tmp/ticket13_validation/revision_wheels/gradient_vanishing_research-0.1.0-py3-none-any.whl
```

The second wheel contained `gradient_vanishing/_build_info.py` with
`SOURCE_REVISION = 'e9a40fc94c1c7bf76d85b2e2ecf4c6691475d2f5'`.
From `C:\Users\Admin`, this command completed and its manifest recorded the
same commit, status `complete`, reduced scope, and three hashed artifacts:

```powershell
& 'D:\ILPT\Project\Gradient-Vanishing\.tmp\ticket13_validation\venv2\Scripts\gradient-vanishing.exe' gradients --config 'D:\ILPT\Project\Gradient-Vanishing\configs\ticket13_reduced_cpu.json' --output-dir 'D:\ILPT\Project\Gradient-Vanishing\.tmp\ticket13_validation\revision_run'
```

The resulting manifest is under
`.tmp/ticket13_validation/revision_run/gradients-20261014T132728-1791984448894931300-9148/`.

From `C:\Users\Admin`, outside the checkout, the installed package imported
from `venv2/Lib/site-packages`; `load_config()` read the bundled default JSON
and returned lengths `[10, 30, 100]`. Both
`venv2/Scripts/gradient-vanishing.exe --help` and
`venv2/Scripts/python.exe -m gradient_vanishing --help` succeeded.

## Reduced configuration and executed commands

The committed [CPU configuration](../configs/ticket13_reduced_cpu.json) was
the exact input for the two pipelines. Its SHA-256 at execution was
`407443266a734f5fcc75eb516bb74cad451d38284903449a0a2e7d146b37fd66`.
The table makes its reduction explicit:

| Dimension | Executed |
| --- | --- |
| Models / length | Vanilla RNN, LSTM, custom GRU; `T=10` |
| Main seeds / conditions | `[0, 1]`; default and tuned; 12 final training runs |
| Optuna | 3 studies, 2 trials per study, 2 updates per trial; original TPE/median-pruner definitions with reduced configured choices |
| Final training | 3 updates, batch 8, validation every update, evaluation size 32; gradient captures at zero-based updates 0 and 2 |
| Initial-gradient diagnostics | 3 models × 2 seeds, float64 |
| Gate ablation | 7 notebook conditions × 2 training seeds, with 2 initial-gradient seeds per condition |
| Numeric sweep | LSTM and custom GRU; biases `[-1, 0, 1]`, 2 initial-gradient seeds each; no training/test MSE per bias |

The wheel-installed commands below were run **twice** for each experiment
stage, from outside the checkout with absolute paths. Each invocation created
a new directory under `runs`; no result was overwritten:

```powershell
$cli='D:\ILPT\Project\Gradient-Vanishing\.tmp\ticket13_validation\venv2\Scripts\gradient-vanishing.exe'
$config='D:\ILPT\Project\Gradient-Vanishing\configs\ticket13_reduced_cpu.json'
$runs='D:\ILPT\Project\Gradient-Vanishing\.tmp\ticket13_validation\runs'
& $cli main --config $config --output-dir $runs
& $cli main --config $config --output-dir $runs
& $cli gradients --config $config --output-dir $runs
& $cli ablation --config $config --output-dir $runs
& $cli sweep --config $config --output-dir $runs
& $cli gradients --config $config --output-dir $runs
& $cli ablation --config $config --output-dir $runs
& $cli sweep --config $config --output-dir $runs
```

The paired run directories were:

| Stage | First directory under `.tmp/ticket13_validation/runs/` | Second directory |
| --- | --- | --- |
| Main | `main-20261014T130637-1791983197595605500-16452/` | `main-20261014T130646-1791983206901307400-3452/` |
| Gradients | `gradients-20261014T130659-1791983219234262900-31052/` | `gradients-20261014T130710-1791983230363636300-2400/` |
| Ablation | `ablation-20261014T130659-1791983219233946000-7200/` | `ablation-20261014T130710-1791983230352301000-13580/` |
| Sweep | `sweep-20261014T130659-1791983219277671900-13868/` | `sweep-20261014T130710-1791983230367523200-10996/` |
| Report | `report-20261014T130730-1791983250775381400-3364/` | `report-20261014T130741-1791983261038394000-21508/` |

### Paired validation after the provenance correction

Both complete reduced pipelines were repeated with the corrected wheel from
`C:\Users\Admin`. The commands were identical to the experiment and report
commands above, with `$runs` set to
`D:\ILPT\Project\Gradient-Vanishing\.tmp\ticket13_validation\revision_pipelines`.
The paired stage commands were invoked in this order:

```powershell
foreach ($repeat in 1..2) {
  foreach ($stage in @('main','gradients','ablation','sweep')) {
    & $cli $stage --config $config --output-dir $runs
    if ($LASTEXITCODE -ne 0) { throw "Failed: $stage repeat $repeat" }
  }
}
```

For each repeat, the `report` command above used that repeat's main, gradient,
ablation, and sweep result paths. Both report commands completed. The new
directories under `revision_pipelines/` are:

| Stage | First run | Second run |
| --- | --- | --- |
| Main | `main-20261014T133156-1791984716055634700-29056/` | `main-20261014T133236-1791984756783366100-19404/` |
| Gradients | `gradients-20261014T133205-1791984725313986000-428/` | `gradients-20261014T133241-1791984761257383900-2248/` |
| Ablation | `ablation-20261014T133218-1791984738098763500-4560/` | `ablation-20261014T133245-1791984765454103700-6452/` |
| Sweep | `sweep-20261014T133230-1791984750020704000-28764/` | `sweep-20261014T133250-1791984770184281700-5932/` |
| Report | `report-20261014T133307-1791984787629976900-29984/` | `report-20261014T133310-1791984790444528400-21248/` |

All ten new manifests recorded status `complete`, identical effective
configuration within each pair, matching artifact SHA-256 hashes, and
`git_commit: e9a40fc94c1c7bf76d85b2e2ecf4c6691475d2f5`. All paired
scientific result records matched exactly after normalizing generated Optuna
study names. Eleven of twelve report products were byte-identical; only the
generated study-name column in `optuna_trials.csv` differed. These are the
complete provenance-backed pipeline pairs. The earlier pairs remain useful
repeatability evidence with incomplete source-revision provenance.

Each experiment manifest reported `complete`, `reduced_or_modified`, CPU,
the actual stage seeds/models/lengths, effective config, Python/dependency
versions, timestamps, and matching SHA-256 hashes for every saved result.
The main result saved `main.json` and three separate study records. The
diagnostic stage saved three model records. Main, diagnostics, ablation, and
sweep saved real numerical outputs; no synthetic records were used here.

## Independent reporting and numeric checks

`gradient-vanishing report` was run twice, once from each result set, with
`--source main=<main.json>`, `study=<study_vanilla_rnn_T10.json>`,
`ablation=<gate_ablation.json>`, `sweep=<retention_bias_sweep.json>`, and all
three repeated `initial_gradients=<model JSON>` arguments. These are the
documented Ticket 12 arguments; no experiment command was invoked by the
report stage. The first report's exact input paths are in its
`reports/report_index.json`. Each report has seven real source records,
`main_summary.csv`, `gate_ablation.csv`, `retention_bias_values.csv`,
`optuna_trials.csv`, four PNG/SVG figure pairs, and `report_index.json`.
There were no skipped kinds. The first report's manifest lists 13 artifacts
including the index. Input hashes matched the result files before and after
reporting. Representative PNG figures were visually inspected for readable
labels and the expected logarithmic validation scale and sweep axes.

For the first report, the exact source selection was:

```powershell
$m=Join-Path $runs 'main-20261014T130637-1791983197595605500-16452/results'
$g=Join-Path $runs 'gradients-20261014T130659-1791983219234262900-31052/results'
$a=Join-Path $runs 'ablation-20261014T130659-1791983219233946000-7200/results'
$s=Join-Path $runs 'sweep-20261014T130659-1791983219277671900-13868/results'
& $cli report --config $config --output-dir $runs `
  --source "main=$(Join-Path $m 'main.json')" `
  --source "study=$(Join-Path $m 'study_vanilla_rnn_T10.json')" `
  --source "ablation=$(Join-Path $a 'gate_ablation.json')" `
  --source "sweep=$(Join-Path $s 'retention_bias_sweep.json')" `
  --source "initial_gradients=$(Join-Path $g 'initial_gradient_vanilla_rnn_T10.json')" `
  --source "initial_gradients=$(Join-Path $g 'initial_gradient_lstm_T10.json')" `
  --source "initial_gradients=$(Join-Path $g 'initial_gradient_gru_T10.json')"
```

The second report used the second-column run directories with the same file
names and arguments.

Direct source-to-output checks found:

| Output | Source calculation and observed report value | Status |
| --- | --- | --- |
| Main Vanilla RNN `T=10` | Source per-seed test MSE gives default `0.3754 ± 0.0419`, tuned `0.6000 ± 0.2572` with `ddof=1`; CSV matches | PASS |
| Gate ablation Vanilla RNN | Source test MSE and initial-gradient decay give `0.3754 ± 0.0419` and `0.437 ± 0.060`; CSV matches | PASS |
| Sweep LSTM at `b=-1` | Source initial-gradient decay `0.3553752939454662`, theoretical retention `0.2689414213699951`; CSV matches | PASS |
| Training heatmap | Source raw captured norms at update 0 yield first displayed log10 ratio `-3.1894547154575488`; the saved heatmap calculation matches | PASS |
| Optuna | Two trial rows for the selected Vanilla RNN study; states, parameters, objective values, and best-trial flag match its JSON | PASS |

Across the two pipelines, every scientific result record matched under the
approved float32 tolerance (`rtol=1e-5`, `atol=1e-6`); in this pinned CPU run
all compared values, orderings, and recorded Torch RNG digests matched
exactly. Only Optuna-generated study names differed. Eleven of twelve report
products were byte-identical; `optuna_trials.csv` differed only in its `Study`
name column, with identical numeric trial values. No numerical discrepancy
was found. This is **same-implementation repeatability**, not independent
notebook equivalence. In this reduced run, tuned models did not consistently
beat defaults; no performance claim follows from three updates.

## Configuration, failure, and source checks

| Objective / procedure | Expected | Observed | Status / evidence |
| --- | --- | --- | --- |
| Default config from installed wheel outside checkout | Bundled notebook defaults load | Lengths `[10,30,100]` and CLI help worked | PASS; wheel import and help commands above |
| Partial JSON with repeated CLI overrides | CLI values take precedence and reach computation | Relative-path `gradients --config configs/ticket13_reduced_cpu.json --set gradients.initial_seeds=3 --set training.batch_size=4 --output-dir .tmp/ticket13_validation/relative_runs` recorded batch 4 and seeds `[0,1,2]`; record has three curves | PASS; `relative_runs/.../manifest.json` and model JSON |
| Absolute paths and existing output root | New run directory each time | Two complete reduced pipelines coexist; prior file hashes unchanged | PASS; run directories above |
| Invalid JSON | Nonzero exit before run creation | `main --config .tmp/ticket13_validation/bad.json` failed; no `bad_runs` directory | PASS; CLI stderr and path check |
| Unsupported stage | Nonzero exit with choices | `gradient-vanishing unsupported` failed | PASS; CLI stderr |
| Full-size safeguard | No default matrix without explicit flag | `main --output-dir .tmp/ticket13_validation/full_guard_runs` failed; no directory | PASS; CLI stderr and path check |
| Missing report source | Nonzero exit, failed manifest, prior results retained | `report --source main=<missing.json>` returned failure; manifest has `ReportInputError`, prior main hash still matched | PASS; failed report run manifest |
| Partial experiment record | Preserve valid prior conditions and status | Existing focused CLI tests inject an interrupted experiment and verify serialized partial results | PASS (test seam); `tests/test_cli.py` |
| Atomic strict JSON | No truncated/invalid result values or RNG change | Existing persistence tests verify replacement, exact float round-trip, explicit nonfinite string, and Torch RNG preservation | PASS (test seam); `tests/test_cli.py` |
| Test-split isolation in tuning | Objective uses validation only | Focused real objective test runs with no test split available | PASS (test seam); `tests/test_experiments.py::test_real_objective_never_uses_test_split` |
| Notebook/fixture integrity | Protected SHA-256 unchanged | Both hashes in the environment table match the pre-implementation values | PASS; SHA-256 checks and regression suite |

The original source cell coverage map remains the one in
[refactoring_spec.md](refactoring_spec.md): installation/setup cells 0–5,
data 6–7, models/init 8–11, training/evaluation 12–13, tuning 14–15,
main comparison 16–21, initial diagnostics 22–23, heatmap 24–25, and
ablation/sweep 26–28. No executable analysis cell was removed by Ticket 13.

## Reproducibility matrix

| Area | Evidence level | Status |
| --- | --- | --- |
| Fresh pinned installation | 38/38 packages, wheel import outside checkout, CLI and package resource | PASS |
| Configuration | Default, partial file, repeated overrides, relative/absolute paths, invalid cases | PASS |
| Data generation and model equations | Immutable fixture and direct-notebook regression tests | PASS at fixture cases |
| Seeded training | Three direct notebook training probes plus repeated reduced main histories | PASS at tested cases |
| Initial and trained hidden gradients | Fixture/direct-notebook tests plus repeated reduced records and heatmap | PASS at tested cases |
| Optuna | Direct-notebook reduced study regression and two repeated real studies per model | PASS at tested cases |
| Main comparison | Direct-notebook reduced comparison regression and two repeated CLI grids | PASS at tested cases |
| Gate ablation | Direct-notebook reduced ablation regression and two repeated CLI runs | PASS at tested cases |
| Retention-bias sweep | Direct-notebook reduced sweep regression and two repeated CLI runs at three biases | PASS at tested cases |
| Reporting | Two saved-result-only runs, source hash and selected numeric/visual checks | PASS at tested cases |
| Run manifests | Stage-specific provenance, status, artifact hashes; two corrected-wheel complete pipeline pairs record `e9a40fc` | PASS for corrected-wheel pairs; PARTIAL for preliminary ten runs |
| Baseline regression | Immutable fixture and full test suite, including direct source execution | PASS at recorded cases |
| Full notebook experiment matrix | Ticket 14 work; not executed here | **NOT VERIFIED** |
| GPU execution and cross-device identity | CPU-only environment; no GPU run | **NOT VERIFIED** |

The known GRU wording, evaluation-mode side effect, and missing learning-curve
uncertainty band remain documented in the approved specification and were not
changed. The notebook's historical saved outputs remain unverified observations.

## Automated acceptance gates

The focused integration test was run both in the existing pinned environment
and in the fresh wheel installation. In the fresh environment, the exact
commands and outcomes were:

```powershell
.tmp\ticket13_validation\venv2\Scripts\python.exe -m pytest -q tests/test_end_to_end.py
# 1 passed
.tmp\ticket13_validation\venv2\Scripts\python.exe -m pytest -q
# 136 passed, 0 failed
```

The full suite includes immutable fixture comparisons, direct execution of
selected original notebook definitions, split-isolation checks, CLI
failure/partial-result tests, and reporting regression tests. Python
`compileall` passed for the package and new integration test. No static type
checker is configured in the pinned environment. The four documented relative
CLI experiment commands in the README were also executed successfully from
the checkout against the committed reduced config; the report command was
executed with the documented `--source KIND=PATH` syntax from outside it.
