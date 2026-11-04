# 開發路線圖

> 優先順序為已確認需求。各階段「驗收標準」是進入下一階段的條件。

## 1. 階段總覽

| 階段 | 項目 | 優先級 | 依賴 | 狀態 |
|---|---|---|---|---|
| 0 | 專案骨架與文件 | — | — | 本次完成 |
| 1 | Core 資料結構與 Screening | 高 | 0 | Planned |
| 2 | RSM Design 與 I-optimal 整合 | 高 | 1 | Planned |
| 3 | **Module 1：Active Learning** | **最高** | 1、2 | Planned |
| 4 | RSM 建模、ANOVA、Stepwise、診斷 | 高 | 1 | Planned |
| 5 | Module 2：殘差修正 | 高 | 4（可獨立於 3） | Planned |
| 6 | **四組研究 Benchmark** | **最高** | 2、3、4、5 | Planned |
| 7 | 最佳化與敏感度分析 | 後期 | 4、5（已驗證模型） | Planned |

## 2. 依賴規則

- RSM Design 提供初始設計給 Module 1。
- Module 1 **不得**依賴最終 RSM 建模；初始使用 GPR。
- 最終 RSM 建模須在 Module 2 整合進最終 hybrid 流程**之前**完成。
- Module 2 **不是** Module 1 的前置條件。
- Benchmark 整合其餘研究模組的輸出。
- Optimization 作用於已驗證的最終預測模型。

## 3. 各階段內容、測試與驗收

### Phase 0 — 骨架與文件（本次）

- 交付：`pyproject.toml`、`src/adore/__init__.py`、smoke test、README、`docs/*`。
- 測試：`pytest`（僅驗證可 import 與版本字串）。
- 驗收：文件術語一致；沒有宣稱未實作的功能。

### Phase 1 — Core 與 Screening

- 內容：Factor / Response / ExperimentDataset / Experiment；natural ↔ coded 轉換；PB、Fractional Factorial、DSD；主效應 / Pareto / half-normal。
- 測試：編碼往返無誤差；資料集各響應共用同一 X；輸入驗證（範圍、重複名稱）；PB / FF 的正交性與已知設計矩陣對照。
- 驗收：核心介面定案並寫入 `architecture.md`；Screening 結果對照教科書範例。

### Phase 2 — RSM Design 與 I-optimal

- 內容：CCD；I-optimal（主要）、D-optimal（比較）；augmentation；候選集與約束區域；秩 / 條件數檢查。
- 測試：CCD 的點數與結構對照公式；I-/D-criterion 對已知小例子的手算值；不可逆設計明確報錯；同 seed 可重現；約束區域內的點全部可行。
- 驗收：I-criterion 近似誤差檢查通過；文件明載 §4.2 所列全部規格；**未**把 GPR 變異稱為 I-optimality。

### Phase 3 — Module 1：Active Learning（最高優先）

- 內容：GPR surrogate（每響應一個）、候選點、acquisition（Max Variance、IVR、I-optimal augmentation）、多響應正規化與聚合、停止條件、序貫迴圈、固定評估集。
- 測試：
  - 停止條件：個別門檻 AND 聚合門檻；負 $R^2$ 保留；SST = 0 時回報原因並視為未達標；預算用盡回報 `budget_exhausted`。
  - 無資料洩漏：評估集與訓練資料不重疊。
  - 可重現：相同 seed → 相同建議點序列。
  - 獨立性：`active_learning` 不 import `rsm`、`residual`（以 import 檢查測試）。
  - 多響應：各響應尺度相差數量級時，聚合結果不被單一響應主導。
- 驗收：在已知解析函數上，至少一種全域導向策略優於隨機 / 單純最大變異 baseline（需多 seed 統計），或明確記錄未優於的結果。

### Phase 4 — 最終 RSM 建模

- 內容：一階 / 二階多項式、stepwise、AICc / BIC、hierarchy、ANOVA、Lack-of-Fit、模型統計量、殘差診斷。
- 測試：係數 / ANOVA / LOF 對照教科書範例與 R 參考實作（`rsm`）；hierarchy 約束成立；明確方程式輸出。
- 驗收：數值差異在記錄的容許誤差內；**通過後**才可在文件中陳述對照結果；不宣稱商業軟體相容。

### Phase 5 — Module 2：殘差修正

- 內容：OOF CV（折內重做模型選擇）、GPR / ANN / RF / SVR 殘差模型、hybrid。
- 測試：殘差目標確由 OOF 預測產生（以構造性測試驗證：改動某觀測值不影響其自身的 OOF 預測）；獨立驗證資料不參與擬合；可選「不修正」退回。
- 驗收：在已知含結構性殘差的資料上，hybrid 的獨立驗證誤差下降；在純噪音資料上不顯著變差。

### Phase 6 — 四組 Benchmark（最高優先）

- 內容：A–D 組、固定預算、目標準確度、指標、多 seed 彙整。
- 測試：各組共用相同 initial design / validation set / 預算；未達標被正確設限並回報；mean ± std 計算。
- 驗收：先定案 §9.3 的 baseline 定義；結果表含未達標比例。

### Phase 7 — 最佳化與敏感度（後期）

- 內容：GA、NSGA-II、SLSQP、Desirability；敏感度分析；外插警告。
- 驗收：僅作用於 Phase 4/5 驗證過的模型。

## 4. 延後項目

- 報告輸出（HTML / PDF / Excel）。
- 資料轉換 / WLS 等 RSM 假設補救（舊規劃有，與 ML 殘差修正是不同概念；需求確認後再排入 Phase 4 後）。
- GA 整合進 Active Learning（除非有明確研究理由）。
- 非 GPR 的 surrogate。
- 完整 Pipeline / YAML 設定。

## 5. 下一步建議

進入 **Phase 1**。需要先決定的事項：座標編碼慣例（coded [−1, 1]）、`ExperimentDataset` 欄位、
Screening 是否沿用 `pyDOE3` 還是自行實作（見 `algorithm_decisions.md` ADR-008）。
