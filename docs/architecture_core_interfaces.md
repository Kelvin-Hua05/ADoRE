# Phase 1：核心介面（定案）

> 這份文件是 `architecture.md` §5「核心資料契約」在 Phase 1 的定案內容，請併入 repo 的 `architecture.md`。

## 決定的事項（Roadmap §5 的待決項目）

| 項目 | 決定 |
|---|---|
| 座標編碼 | coded ∈ [−1, +1]：`coded = (x − (high+low)/2) / ((high−low)/2)`；中心點 = 0。類別型因子只有 ±1。設計若需超出區間（如外接 CCD），資料集 `info["allow_outside"]=True` 才允許 |
| `ExperimentDataset` 欄位 | 下表 |
| Screening 實作 | 自行實作 PB / FF（ADR-008：沙箱無 pyDOE3、無網路）；OLS 以 NumPy 實作 |

## 契約

| 物件 | 欄位 / 行為 |
|---|---|
| `Factor(name, low, high, unit, factor_type, metadata)` | `to_coded` / `from_coded`；`low < high`、有限、名稱非空；`numeric` 或 `categorical`（兩水準） |
| `Response(name, weight, r2_threshold, direction, target, unit)` | 驗證：`weight ≥ 0`、`r2_threshold ≤ 1`、`direction ∈ {None, maximize, minimize, target}` |
| `ExperimentDataset` | 一個 coded 矩陣 `X`（n×k）**由所有響應共用**；每個響應一欄（NaN = 尚未量測）；`std_order`（= `run_id`，列識別，不變）與 `run_order`（執行順序，排列）分離；`point_type`、`source`（`initial` / `AL-iter-k` / `validation` …）、`stage`；`info`（design_type、seed、模型規格…）；`response_specs`（各響應的 `Response`） |
| 操作 | `add_response(name, values, order="std"\|"run")`、`append_runs(coded, source, values)`（新 run 排在既有 run 之後）、`subset` / `copy`、`training()` / `validation()`、`replicate_groups()` / `n_unique_points` / `n_replicated_runs`、`validate()`、`to_csv` / `from_csv`、`randomize(seed)`、`experiment(run_id) -> Experiment` |
| `Experiment` | 單一 run 的唯讀檢視（coded、actual、responses、point_type、source、stage） |

## 不變條件（有測試）

* 所有響應與同一個 X 對齊；長度不符的響應會被拒絕。
* `append_runs` 後既有響應補 NaN、`run_order` 延續、`source` / `stage` 記錄新 run 的來源。
* CSV 往返無損（含 source、stage、response 設定、執行順序、dummy 欄位）；actual 與 coded 欄位不一致時報錯。
* 驗證資料（`source="validation"`）可由 `training()` 排除，不會混入訓練。
* 所有隨機程序（`randomize`、設計產生）接受顯式 seed 並記錄於 `info["seed"]`。
