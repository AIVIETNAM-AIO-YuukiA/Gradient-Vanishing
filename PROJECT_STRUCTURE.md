# Cấu Trúc Chi Tiết Dự Án (Project Structure & Git Status)

Tài liệu này cung cấp cái nhìn toàn diện về cấu trúc thư mục, chức năng của từng tệp tin và trạng thái Git tương ứng trong dự án **Gradient-Vanishing** (chuyển đổi và tái hiện mã nguồn từ notebook nghiên cứu hiện tượng Vanishing Gradient sang gói Python chuẩn).

---

## 1. Thông Tin Nhánh & Tổng Quan Trạng Thái Git

* **Nhánh hiện tại (Current Branch):** `refactor/notebook-to-project`
* **Trạng thái môi trường:** Python 3.13 + PyTorch 2.11.0 (CPU) + Optuna 5.0.0
* **Kết quả kiểm thử:** 136/136 tests passed (`pytest tests/`)

### Bảng tóm tắt phân loại trạng thái Git:

| Trạng thái (Status) | Ký hiệu | Ý nghĩa | Số lượng tệp chính |
|:---|:---:|:---|:---:|
| **Tracked (Modified)** | `M` | Tệp đã được Git theo dõi và vừa được chỉnh sửa (chưa commit) | 6 tệp |
| **Untracked** | `??` | Tệp mới tạo trên đĩa, chưa được thêm vào Git | Các thư mục `runs/`, `.idea/`, `.scratch/spec.md`, các tài liệu agent |
| **Tracked (Unmodified)** | ` ` | Tệp đang được theo dõi, sạch (clean), không có thay đổi | Toàn bộ source code, docs, test gốc |

---

## 2. Danh Sách Tệp Đang Thay Đổi (Modified Files)

Những tệp này vừa được điều chỉnh để đồng bộ với phiên bản **Optuna 5.0.0** (phiên bản mà notebook gốc sử dụng), giải quyết triệt để vấn đề sai lệch kết quả giữa notebook và gói code:

| Đường dẫn tệp | Trạng thái | Mô tả chi tiết thay đổi |
|:---|:---:|:---|
| [`pyproject.toml`](pyproject.toml) | `M` (Modified) | Nâng version dependency `optuna==4.8.0` $\rightarrow$ `optuna==5.0.0` để khớp với notebook. |
| [`requirements-cpu.lock`](requirements-cpu.lock) | `M` (Modified) | Khóa phiên bản gói cài đặt CPU tham chiếu sang `optuna==5.0.0`. |
| [`tests/fixtures/notebook_baseline.json`](tests/fixtures/notebook_baseline.json) | `M` (Modified) | Cập nhật phiên bản Optuna trong metadata môi trường, hash `environment_sha256` của 22 cases kiểm thử và `integrity_sha256` của toàn bộ tài liệu. |
| [`tests/test_experiments.py`](tests/test_experiments.py) | `M` (Modified) | Cập nhật chuỗi SHA-256 tham chiếu của file fixture `notebook_baseline.json`. |
| [`tests/test_gradients.py`](tests/test_gradients.py) | `M` (Modified) | Cập nhật chuỗi SHA-256 tham chiếu của file fixture `notebook_baseline.json`. |
| [`tests/test_training.py`](tests/test_training.py) | `M` (Modified) | Cập nhật chuỗi SHA-256 tham chiếu của file fixture `notebook_baseline.json`. |

---

## 3. Cây Thư Mục Toàn Diện & Trạng Thái Chi Tiết Của Từng Tệp

### 3.1. Thư mục gốc (Root Directory)

```text
Gradient-Vanishing/
├── .gitignore                          [Tracked]    Cấu hình bỏ qua các tệp tạm, venv, cache, build
├── setup.py                            [Tracked]    Tương thích cài đặt truyền thống (gọi setuptools)
├── pyproject.toml                      [Modified]   Khai báo metadata, build-system, dependencies, CLI script
├── requirements-cpu.lock               [Modified]   Snapshot danh sách các thư viện CPU đã được xác thực
├── README.md                           [Tracked]    Hướng dẫn tổng quan về dự án, cách cài đặt và chạy
├── CONTEXT.md                          [Untracked]  Tài liệu ngữ cảnh về kiến trúc và miền bài toán
├── AGENTS.md                           [Untracked]  Tài liệu hướng dẫn kỹ năng cho các AI Coding Agents
```

---

### 3.2. Mã nguồn chính (`src/gradient_vanishing/`)

Mã nguồn được đóng gói dạng module chuẩn theo cấu trúc `src/`, bảo đảm tính module hóa, kiểm soát chặt chẽ việc gọi chéo và RNG determinism:

```text
src/gradient_vanishing/
├── __init__.py                         [Tracked]    Khởi tạo package gradient_vanishing
├── __main__.py                         [Tracked]    Entrypoint cho lệnh chạy module: python -m gradient_vanishing
├── cli.py                              [Tracked]    Giao diện dòng lệnh CLI (main, gradients, ablation, sweep, report)
├── config.py                           [Tracked]    Đọc, kiểm tra tính hợp lệ (validation) cấu hình JSON và thiết lập runtime CUDA/CPU
├── data.py                             [Tracked]    Sinh dữ liệu bài toán ghi nhớ phần tử đầu tiên (First-element memorization) & cache tập val/test
├── models.py                           [Tracked]    Cài đặt 3 mô hình từ số 0 (Vanilla RNN, LSTM, GRU) kèm cơ chế track gradient từng bước thời gian
├── training.py                         [Tracked]    Vòng huấn luyện tất định có seed (deterministic training), tracking loss, tích hợp Optuna Pruner
├── gradients.py                        [Tracked]    Đo lường độ lớn norm gradient của hidden states, tính hệ số suy giảm (decay factor)
├── experiments.py                      [Tracked]    Điều phối 4 thí nghiệm chính: Optuna tuning, Main comparison, Gate ablation, Retention sweep
├── reporting.py                        [Tracked]    Render kết quả dạng bảng CSV và vẽ đồ thị (learning curves, heatmaps) từ file JSON có sẵn
├── run_manifest.py                     [Tracked]    Tạo thư mục run định danh theo thời gian và tính toán hash SHA-256 đảm bảo tính toàn vẹn (provenance)
└── configs/
    └── default.json                    [Tracked]    Cấu hình thực nghiệm mặc định tương đương 100% notebook gốc
```

#### Chi tiết vai trò từng module trong `src/`:
* **`models.py`**: Cài đặt thủ công (`VanillaRNN_Tracker`, `LSTM_Tracker`, `GRU_Tracker`) kế thừa `TrackerBase` để gọi `h.retain_grad()` nhằm giữ lại gradient của trạng thái ẩn trung gian sau `backward()`. Bao gồm hàm khởi tạo trọng số (`orthogonal`, `xavier`) và bias cổng (`zero`, `one`, `chrono`).
* **`data.py`**: Sinh tensor chuỗi $X \in \mathbb{R}^{T \times B \times 1}$ với chỉ bước $t=0$ mang giá trị ngẫu nhiên $\sim \mathcal{U}[-1, 1]$, các bước còn lại bằng 0; nhãn $Y = X[0]$.
* **`experiments.py`**:
  * `tune_model`: Tìm siêu tham số tối ưu (`lr`, `hidden_size`, `init_method`) bằng `TPESampler` và `MedianPruner`.
  * `run_main_comparison`: Chạy so sánh theo lưới $T \in \{10, 30, 100\}$, 3 mô hình, 2 cấu hình (default & tuned), 5 seeds.
  * `run_gate_ablation`: Đánh giá 7 trường hợp khởi tạo bias cổng ở $T=100$.
  * `run_retention_bias_sweep`: Quét liên tục 17 mức bias từ -4.0 đến +4.0.

---

### 3.3. Notebook nghiên cứu gốc (`Notebooks/`)

```text
Notebooks/
└── GradientVanishingAnalysis.ipynb     [Tracked]    File Jupyter Notebook gốc tham chiếu bất biến (SHA-256: 71ec17cd...)
```

---

### 3.4. Bộ kiểm thử tự động (`tests/`)

Toàn bộ logic được bảo vệ bởi 136 kiểm thử hồi quy nghiêm ngặt:

```text
tests/
├── fixtures/
│   └── notebook_baseline.json          [Modified]   Dữ liệu kết quả chạy trực tiếp từ notebook dùng làm oracle kiểm thử
├── test_config.py                      [Tracked]    Kiểm thử cấu hình JSON, các ràng buộc hợp lệ và tham số ghi đè
├── test_data.py                        [Tracked]    Kiểm thử tính tất định và phân phối của tập dữ liệu sinh ra
├── test_models.py                      [Tracked]    Kiểm thử kiến trúc RNN/LSTM/GRU và các chế độ khởi tạo cổng
├── test_training.py                    [Modified]   Kiểm thử tính tái hiện vòng huấn luyện và đánh giá mô hình
├── test_gradients.py                   [Modified]   Kiểm thử đo lường gradient trạng thái ẩn và tính toán độ suy giảm
├── test_notebook_baseline.py           [Tracked]    Kiểm thử đối chiếu từng case độc lập với oracle notebook_baseline.json
├── test_experiments.py                 [Modified]   Kiểm thử luồng Optuna study và tính lặp lại của TPE sampler
├── test_main_comparison.py             [Tracked]    Kiểm thử cấu trúc kết quả thí nghiệm chính và bảng tóm tắt
├── test_gate_ablation.py               [Tracked]    Kiểm thử logic thí nghiệm cắt bỏ (ablation) bias cổng
├── test_retention_bias_sweep.py        [Tracked]    Kiểm thử quét liên tục bias lưu giữ (retention bias sweep)
├── test_reporting.py                   [Tracked]    Kiểm thử kết xuất báo cáo, bảng biểu CSV và biểu đồ PNG/SVG
├── test_cli.py                         [Tracked]    Kiểm thử giao diện dòng lệnh CLI và tham số `--allow-full`
└── test_end_to_end.py                  [Tracked]    Kiểm thử tích hợp luồng đầu cuối (end-to-end pipeline)
```

---

### 3.5. Cấu hình kiểm thử và thực nghiệm (`configs/`)

```text
configs/
└── ticket13_reduced_cpu.json           [Tracked]    File cấu hình rút gọn để chạy kiểm tra nhanh phần mềm trên CPU
```

---

### 3.6. Tài liệu kỹ thuật & Báo cáo tái hiện (`docs/`)

Hệ thống tài liệu hoàn chỉnh ghi nhận toàn bộ quá trình tái cấu trúc:

```text
docs/
├── architecture_proposal.md            [Tracked]    Đề xuất kiến trúc module hóa
├── baseline_fixtures.md                [Tracked]    Phương pháp trích xuất dữ liệu oracle từ notebook
├── commands.md                         [Tracked]    Hướng dẫn chi tiết các lệnh CLI và cờ an toàn
├── data.md                             [Tracked]    Đặc tả bài toán ghi nhớ dữ liệu
├── end_to_end_validation.md            [Tracked]    Báo cáo kiểm thử tích hợp rút gọn
├── full_experiment_execution_plan.md   [Tracked]    Kế hoạch chạy toàn bộ lưới thực nghiệm
├── full_reproducibility_report.md      [Tracked]    Báo cáo chi tiết về mức độ tái hiện và các so sánh lịch sử
├── gate_ablation.md                    [Tracked]    Tài liệu giải thích thí nghiệm ablation bias
├── gradient_diagnostics.md             [Tracked]    Định nghĩa toán học của gradient trạng thái ẩn & decay fit
├── installation.md                     [Tracked]    Hướng dẫn cài đặt môi trường độc lập
├── main_comparison.md                  [Tracked]    Thiết kế thí nghiệm so sánh chính
├── optuna_study.md                     [Tracked]    Không gian tìm kiếm và chiến lược pruning của Optuna
├── refactoring_spec.md                 [Tracked]    Đặc tả kỹ thuật tái cấu trúc đã được phê duyệt
├── reporting.md                        [Tracked]    Định nghĩa các bảng và biểu đồ kết quả
├── retention_bias_sweep.md             [Tracked]    Đặc tả quét numeric bias
├── notebook_audit.md                   [Untracked]  Biên bản kiểm tra toàn bộ tế bào (cell) của notebook gốc
├── ticket_plan.md                      [Untracked]  Kế hoạch phân bổ nhiệm vụ theo từng ticket
├── agents/                             [Untracked]  Các tài liệu quy chuẩn dành cho Agent
│   ├── domain.md
│   ├── issue-tracker.md
│   └── triage-labels.md
└── followups/                          [Untracked]  Các ghi chú phân tích chuyên sâu
    ├── 01-gru-gate-terminology.md      Lưu ý về quy ước dấu bias trong GRU tự cài đặt
    ├── 02-evaluate-mode.md             Lưu ý về tác dụng phụ (side-effect) chế độ train/eval
    └── 03-learning-curve-uncertainty.md Đề xuất dải độ lệch chuẩn cho đồ thị học
```

---

### 3.7. Kế hoạch theo dõi công việc (`.scratch/`)

```text
.scratch/gradient-vanishing-refactor/
├── map.md                              [Tracked]    Bản đồ liên kết giữa các ticket và artifacts
├── spec.md                             [Untracked]  Bản sao đặc tả chi tiết của dự án
└── issues/                             [Tracked]    14 ticket theo dõi tiến độ từ Ticket 01 đến Ticket 14
    ├── 01-installable-project-and-validated-configuration.md
    ├── 02-capture-immutable-baseline-fixtures.md
    ├── 03-reproduce-first-element-memorization-data.md
    ├── 04-extract-recurrent-models-and-gate-initialization.md
    ├── 05-reproduce-seeded-training-and-evaluation.md
    ├── 06-reproduce-hidden-gradient-diagnostics.md
    ├── 07-reproduce-optuna-study.md
    ├── 08-reproduce-main-comparison.md
    ├── 09-reproduce-gate-initialization-ablation.md
    ├── 10-reproduce-retention-bias-sweep.md
    ├── 11-regenerate-tables-and-figures.md
    ├── 12-standardize-commands-and-run-manifests.md
    ├── 13-validate-end-to-end-and-document-use.md
    └── 14-full-experiment-matrix-report.md
```

---

### 3.8. Dữ liệu thực nghiệm vừa chạy sinh ra (`runs/`)

Thư mục lưu trữ kết quả chạy độc lập, tự động sinh sau khi chạy lệnh CLI `main` và `report`:

```text
runs/
├── main-20261014T162411-1791995051528387400-18468/    [Untracked]
│   ├── manifest.json                                   Metadata chứng thực quá trình chạy (RNG seed, thời gian, commit)
│   └── results/
│       ├── main.json                                   Kết quả số học đầy đủ của 3 độ dài chuỗi, 3 model, 5 seed
│       ├── study_vanilla_rnn_T10.json ...              Kết quả chi tiết từng trial Optuna của Vanilla RNN
│       ├── study_lstm_T10.json ...                     Kết quả chi tiết từng trial Optuna của LSTM
│       └── study_gru_T10.json ...                      Kết quả chi tiết từng trial Optuna của GRU
└── report-20261014T164059-1791996059341184600-25276/  [Untracked]
    ├── manifest.json                                   Metadata chứng thực lần xuất báo cáo
    └── reports/
        ├── main_summary.csv                            Bảng kết quả so sánh MSE tổng hợp (Mặc định vs Tuned)
        ├── main_learning_curves.png (.svg)             Đồ thị đường cong học tập (Learning Curves)
        ├── training_gradient_heatmap_T100.png (.svg)   Heatmap phân tích gradient chuẩn hóa ở T=100
        └── report_index.json                           Mục lục kiểm kê hash của các biểu đồ và bảng
```

---

## 4. Tóm Tắt Khuyến Nghị Thao Tác Git Tiếp Theo

Nếu bạn muốn lưu lại toàn bộ các sửa đổi vừa thực hiện để đồng bộ vĩnh viễn với Optuna 5.0:

```powershell
# 1. Thêm các tệp code và test đã sửa vào staging:
git add pyproject.toml requirements-cpu.lock tests/

# 2. Tạo commit ghi nhận việc đồng bộ Optuna 5.0:
git commit -m "fix(deps): bump optuna to 5.0.0 and synchronize baseline fixture hashes"

# 3. (Tùy chọn) Thêm tài liệu cấu trúc dự án:
git add PROJECT_STRUCTURE.md
git commit -m "docs: add detailed project structure and git status map"
```
