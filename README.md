# ADoRE

**Adaptive Design of Experiments with Residual Enhancement**

整合 **回應曲面法（RSM）**、**主動學習（Active Learning）** 與 **以機器學習進行殘差修正（ML Residual Correction）** 的自適應實驗設計框架。

> ⚠️ **狀態：Pre-alpha（文件與架構階段）**
> 目前**沒有任何研究演算法被實作**。repo 內只有套件骨架（`import adore`、`adore.__version__`）、
> 架構文件與開發規劃。下文所有功能皆標示實作狀態，未標示「已實作」者都不能使用。

---

## 1. 簡介

ADoRE 的目標：

1. 以**更少的昂貴實驗**達到預先設定的預測準確度。
2. 透過序貫式實驗設計（sequential design）提升**全域預測表現**。
3. 以 ML 殘差修正提升 RSM 的預測準確度。
4. 保留傳統 RSM 的**可解釋性**（明確方程式、ANOVA、Lack-of-Fit）。
5. 量化「Adaptive DOE」與「Residual Enhancement」各自及合併的效益。

主要研究應用：半導體製程與 layout 建模，特別是寄生電容、電阻等與設計相關的響應預測。

## 2. 研究動機

傳統 DOE 一次決定整組實驗點，不會根據已得資料調整；傳統 RSM 的多項式模型在曲率複雜時，
殘差中常含有結構性訊號。ADoRE 同時處理這兩件事：實驗點「怎麼選」（Adaptive DOE），
以及模型「怎麼補」（Residual Enhancement），並在相同條件下做公平對照（四組 benchmark）。

## 3. 整體流程

```text
[選用] Screening
        │
        ▼
RSM Design（CCD / BBD / I-optimal / …）──► 初始實驗資料
        │
        ▼
Module 1：Active Learning（GPR 代理模型，可獨立執行）
   訓練 → 候選點 → 全域評估 → acquisition → 建議下一點/批次
   → 取得新實驗結果 → 更新 → 檢查停止條件 → 重複
        │  累積實驗資料
        ▼
最終 RSM 建模（多項式、Stepwise、AICc/BIC、ANOVA、LOF、診斷）
        │
        ▼
Module 2：ML 殘差修正（可在給定 RSM 與資料時獨立執行）
        │
        ▼
Benchmark（四組比較）    最佳化（GA / NSGA-II / SLSQP / Desirability）    敏感度分析
```

詳細圖示見 [`docs/architecture.md`](docs/architecture.md)。

## 4. Module 1 — Active Learning（最高優先）

- 輸入初始 RSM 實驗資料 → 以 **GPR** 為初始代理模型 → 建議下一個實驗點或一批實驗點。
- 目標是 **Global Prediction Improvement**，**不是**單純挑預測不確定度最高的點。
- 預計研究的策略：Maximum Predictive Variance、Integrated Predictive Variance Reduction、
  I-optimal Augmentation，以及其他以全域預測為導向的 acquisition。
- **不依賴**最終 RSM 建模模組或 Module 2。
- **GA 不屬於** Active Learning。

## 5. RSM Design 與 I-optimal 整合

- 初始設計必須是 RSM-based：CCD、Box-Behnken、I-optimal，必要時其他 optimal design。
- Screening 為選用的前置階段；已知關鍵因子時可直接進入 RSM Design。
- **I-optimal 不是獨立的開發階段**，而是整合在 RSM Design、Design Augmentation 與 Sequential Design 中；
  I-optimality 為主要準則，D-optimality 保留作比較。
- 經典線性模型的 I-optimality 與 GPR 預測不確定度**不可視為數學等價**（見 `docs/research_methodology.md` §4）。

## 6. 最終 RSM 建模

一階 / 二階多項式、Stepwise、AICc / BIC、Hierarchy 約束、ANOVA、Lack-of-Fit、模型統計量與殘差診斷。
最終模型必須保有**明確的數學方程式**與統計可解釋性。

## 7. Module 2 — ML 殘差修正

對每個響應：`e_i = y_i − ŷ_RSM,i`，混合預測 `ŷ_final(x) = ŷ_RSM(x) + ê_ML(x)`。

- 候選模型：GPR、ANN、Random Forest、SVR 等；**不預設 ANN 最佳**。
- 使用 cross-validation 時，殘差目標必須由 **out-of-fold（OOF）RSM 預測**建立，避免 in-sample 洩漏。
- 保留獨立驗證 / 確認資料；以**泛化表現**而非訓練殘差下降評估。

## 8. 四組 Benchmark

| 組別 | 實驗設計 | 最終模型 |
|---|---|---|
| A | Traditional DOE | RSM |
| B | Adaptive DOE | RSM |
| C | Traditional DOE | RSM + ML Residual Correction |
| D | Adaptive DOE | RSM + ML Residual Correction |

兩種比較：**固定 run 預算**（如 15 / 20 / 25 / 30 runs；RMSE、MAE、R²）與
**目標準確度**（達標所需 run 數；未達標必須明確回報）。
公平性要求：相同初始條件與初始 run 數、相同獨立驗證集、相同預算、可重現亂數種子、
多種子重複並回報平均與標準差。

## 9. 多響應停止條件

所有響應共用同一組實驗點，但**各自建模、各自評估**。停止需**同時**滿足：

- 每個響應的個別最小預測 R²（例：各 ≥ 0.90），**且**
- 聚合預測 R²（平均或加權平均，例：≥ 0.95），

或達到最大 run 預算（此時必須標示「目標未達成」）。可設定 batch size 與 cross-validation 策略。
訓練 R² 不得作為唯一停止依據；負的 R² 必須保留；無法計算時回報原因並視為未達標。

## 10. 最佳化與敏感度分析

GA 保留給**最終製程 / 設計參數最佳化**（GA、NSGA-II、SLSQP、Desirability Function），
作用於已驗證的最終預測模型。敏感度分析同屬後期項目。兩者目前皆為 Planned。

## 11. 功能與開發狀態

**Implemented**

- 套件骨架：`import adore`、`adore.__version__`（僅 smoke test）

**Partially Implemented**

- （無）

**Planned**（依優先順序）

| 順序 | 項目 | 優先級 |
|---|---|---|
| 1 | Core 資料結構與 Screening | 高 |
| 2 | RSM Design 與 I-optimal 整合 | 高 |
| 3 | Module 1：Active Learning | **最高** |
| 4 | RSM 建模、ANOVA、Stepwise、診斷 | 高 |
| 5 | Module 2：殘差修正 | 高 |
| 6 | 四組研究 Benchmark | **最高** |
| 7 | 最佳化與敏感度分析 | 後期 |

## 12. 安裝與快速開始

目前**尚未發佈到 PyPI**。僅支援從原始碼安裝（Python ≥ 3.10）：

```bash
git clone <your-repo-url>
cd <repo-dir>
pip install -e ".[dev]"
pytest
```

目前唯一可用的 API：

```python
import adore
print(adore.__version__)
```

**以下是概念性流程（pseudocode），所有呼叫都尚未實作，僅用於說明預期用法：**

```text
factors, responses = define(...)
initial = rsm_design(factors, method="i_optimal", n_runs=n0)      # 概念
loop:
    surrogate.fit(dataset)                                        # GPR
    next_points = acquisition.recommend(surrogate, candidates, eval_set, batch_size)
    dataset.add(run_experiment(next_points))
    if stopping_rule.satisfied(dataset): break
rsm_model = rsm.fit(dataset)                                      # Module 1 之後
hybrid = residual.fit(rsm_model, dataset, cv="oof")               # Module 2
```

## 13. 專案結構

目前實際存在：

```text
├── src/adore/__init__.py
├── tests/test_package.py
├── docs/
├── pyproject.toml
└── README.md
```

目標架構（**尚未建立**，各目錄於對應開發階段才建立，不預先建立空模組）：

```text
src/adore/
├── core/ screening/ design/ active_learning/ rsm/
├── residual/ benchmark/ optimize/ sensitivity/
└── pipeline.py
```

模組職責與依賴規則見 [`docs/architecture.md`](docs/architecture.md)。

## 14. 驗證與可重現性

- 目前沒有任何統計或演算法結果可供驗證，**不宣稱**與任何商業軟體相容，也不宣稱「已完整驗證」。
- 規劃的驗證方式：以教科書公開範例與 R 參考實作（`rsm`、`AlgDesign`、`FrF2` 等）做數值回歸測試；
  通過後才會在文件中記載對照結果。
- 所有隨機程序（候選點、評估集、初始設計、CV 切分）都必須接受顯式 seed。

## 15. 開發路線

見 [`docs/development_roadmap.md`](docs/development_roadmap.md)（各階段測試與驗收標準）。

## 16. 限制與未來工作

- 目前僅為架構與文件階段。
- 開放的研究決策（Traditional DOE baseline 定義、多響應正規化方式、停止條件的 CV 偏誤、
  GPR 代理與最終 RSM 之間的模型不匹配）記錄於
  [`docs/research_methodology.md`](docs/research_methodology.md) 與
  [`docs/algorithm_decisions.md`](docs/algorithm_decisions.md)。

## 文件索引

| 文件 | 內容 |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | 架構圖、模組職責、資料流、介面草案 |
| [`docs/research_methodology.md`](docs/research_methodology.md) | 數學目標、策略、停止條件、benchmark 方法 |
| [`docs/development_roadmap.md`](docs/development_roadmap.md) | 階段、依賴、測試、驗收標準 |
| [`docs/algorithm_decisions.md`](docs/algorithm_decisions.md) | ADR 形式的技術決策 |
| [`docs/repository_audit.md`](docs/repository_audit.md) | 舊 repo（DoE-ML-GA / YourPkg）審查與遷移說明 |
