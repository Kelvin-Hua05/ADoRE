# 演算法與技術決策紀錄（ADR）

狀態說明：**Accepted** = 已確認需求；**Proposed** = 本次提出、待驗證或待確認。

---

## ADR-001：I-optimal 整合進 RSM Design 與 Active Learning，而非獨立階段

**狀態**：Accepted

- **Context**：舊規劃以 D-optimal 為主；I-optimal 準則直接對應「全域預測變異」，與本研究目標一致。
- **Decision**：I-optimal 為主要 optimal 準則，整合在 RSM Design、Design Augmentation、Sequential Design；D-optimal 保留作比較。
- **Alternatives**：(1) 獨立的 I-optimal 開發階段；(2) 繼續以 D-optimal 為主。
- **Rationale**：I-optimal 不是獨立工作流，而是同一套「模型 + 區域 + 候選集」機制的準則選項；D-optimal 偏重係數估計精度，不直接對應預測品質。
- **Consequences**：`design` 需要共用的候選集 / 積分點 / 秩檢查機制；Module 1 可呼叫 `design.augmentation`。
- **Open questions**：積分點數 $m$ 的選擇與近似誤差判準；約束區域的取樣方式。

## ADR-002：GPR 作為 Module 1 的初始代理模型

**狀態**：Accepted

- **Context**：初始樣本少（約 10–30 runs），需要可給出預測不確定度的代理模型。
- **Decision**：以 GPR 為初始 surrogate，每個響應各一個。
- **Alternatives**：Random Forest（不確定度估計較粗）、ANN / ensemble（小樣本不穩定）、貝氏多項式回歸（與最終 RSM 較一致）。
- **Rationale**：GPR 在小樣本下給出校準的後驗變異，且固定超參數時可用 rank-1 更新快速計算「加入候選點後」的變異。
- **Consequences**：結果依賴 kernel 與超參數假設；`Surrogate` 介面保持可替換。
- **Open questions**：kernel 選擇（Matérn vs RBF、ARD）；超參數每輪重估或固定；多響應是否共享 kernel。

## ADR-003：Active Learning 的目標是全域預測改善

**狀態**：Accepted

- **Context**：單純挑最大預測變異的點傾向集中在邊界 / 外圍，對全域預測的改善有限。
- **Decision**：策略以評估集上的全域指標（Global RMSE / MAE / predictive R²、平均預測變異）比較；Max Variance 僅作 baseline。
- **Alternatives**：以單點不確定度為準。
- **Rationale**：研究目標是整個設計區域的預測準確度。
- **Consequences**：需要固定評估集與（模擬器 / 資料池情境下的）真值；真實實驗下只能用模型不確定度與 CV 代理，須分開報告。
- **Open questions**：多響應聚合的正規化方式（見研究方法論 §6.1）。

## ADR-004：GA 保留給最終最佳化

**狀態**：Accepted

- **Context**：舊規劃把 GA 列為流程步驟之一；Active Learning 的問題是「下一個實驗做哪裡」，不是黑箱全域搜尋。
- **Decision**：GA / NSGA-II / SLSQP / Desirability 只用於最終製程 / 設計參數最佳化，不進入 Module 1。
- **Alternatives**：以 GA 搜尋 acquisition 最大值。
- **Rationale**：候選集上的 acquisition 評估已足夠，不需要 GA；避免把最佳化與資料蒐集混在一起。
- **Consequences**：`optimize` 在 Phase 7 才開發，只作用於已驗證的最終模型。
- **Open questions**：若候選集過大導致 acquisition 評估成本高，才重新評估（需明確研究理由）。

## ADR-005：Module 1 與 Module 2 彼此獨立

**狀態**：Accepted

- **Context**：Active Learning 必須能在尚無最終 RSM 時運作；殘差修正需要能獨立套用於任何提供的 RSM 與資料。
- **Decision**：二者只透過 `ExperimentDataset` 交接；`active_learning` 不得依賴 `rsm` / `residual`；Module 2 不是 Module 1 的前置條件。
- **Alternatives**：以最終 RSM 的指標驅動 Active Learning。
- **Rationale**：降低耦合，可單獨驗證與單獨使用。
- **Consequences**：GPR 的停止指標與最終 RSM 的表現可能不一致（研究方法論 §11 項 2），須在 Benchmark 階段兩者並報。
- **Open questions**：是否在 benchmark 層（而非 Module 1 內）提供「以 RSM 表現決定停止」的選項。

## ADR-006：殘差修正必須使用 OOF 殘差

**狀態**：Accepted

- **Context**：以訓練集上的 in-sample 殘差訓練殘差模型，殘差被低估，殘差模型學到的是擬合假象。
- **Decision**：殘差目標由 OOF RSM 預測建立（折內重做模型選擇），殘差模型以 OOF 殘差訓練，並在獨立驗證資料評估。
- **Alternatives**：in-sample 殘差（洩漏）；單次 hold-out 切分（小樣本浪費資料）。
- **Rationale**：避免洩漏，估計真實泛化表現。
- **Consequences**：OOF 殘差略大於全資料 RSM 殘差（分布偏差）；小樣本下 CV 方差大，須文件化。
- **Open questions**：小樣本下 LOO 與重複 K-fold 的取捨；是否加入對殘差預測的收縮（shrinkage）。

## ADR-007：Python 套件命名

**狀態**：Proposed（需要專案擁有者確認）

- **Context**：舊 repo 使用 `yourpkg` 佔位名稱。檢查 PyPI（2026-11-04）：`adore` **已被其他專案使用**（"Adaptive Derivative Order Randomized Explanation"）；`adore-doe` 目前查無此名稱（尚未註冊）。
- **Decision**：import 名稱 `adore`；發佈名稱（distribution）暫定 `adore-doe`；repo 名稱另定。三者可以不同。
- **Alternatives**：以 `adore` 發佈（被佔用，不可行）；import 名稱也改成與 PyPI 不衝突的名稱（例如 `adore_doe`）。
- **Rationale**：保留 `import adore` 的簡潔；發佈名稱避開衝突。
- **Consequences**：**風險**：若使用者環境同時安裝了 PyPI 上的 `adore` 套件，兩者都提供頂層模組 `adore`，會互相衝突。若這個風險不可接受，應改 import 名稱。
- **Open questions**：是否註冊 `adore-doe`；是否改 import 名稱。

## ADR-008：依賴策略（最小化、延後決定）

**狀態**：Proposed

- **Context**：舊規劃列出 `statsmodels`、`scikit-learn`、`pyDOE3`、`SALib`、`pymoo`、`DEAP`；其中 `pymoo` 與 `DEAP` 功能重疊（GA / NSGA-II）。
- **Decision**：目前 `pyproject.toml` 只宣告 `numpy`、`scipy`、`pandas`，以及可選的 `ml`（scikit-learn）、`stats`（statsmodels）、`dev`。其餘於對應階段才決定。
- **Alternatives**：先全部裝入。
- **Rationale**：不為尚未實作的功能增加依賴；Phase 7 時在 `pymoo` 與 `DEAP` 之間擇一。
- **Consequences**：Phase 1/2 時需決定 Screening / CCD / BBD 是用 `pyDOE3` 還是自行實作；I-optimal 與 stepwise / LOF 的細節控制多半需要自行實作並對照 R 參考實作。
- **Open questions**：RSM 建模用 `statsmodels` 還是自行以 numpy 實作（影響 AICc、hierarchy stepwise 的可控性）。

## ADR-009：預測 R² 的定義與邊界情況

**狀態**：Proposed

- **Context**：停止條件依賴 $R^2$；不同定義（訓練、CV、hold-out）與邊界情況會改變結論。
- **Decision**：停止條件使用 held-out / OOF 的預測 $R^2 = 1-\mathrm{SSE}/\mathrm{SST}$；負值保留；SST = 0 或樣本數不足時回報原因並視為未達標；訓練 $R^2$ 僅作報告，不單獨用於停止。
- **Alternatives**：Adjusted $R^2$、PRESS-based predicted $R^2$（以 LOO 為特例）。
- **Rationale**：反映泛化而非擬合。
- **Consequences**：序貫選點下的 CV 估計可能有偏（研究方法論 §11 項 1）。
- **Open questions**：SST 以 OOF 池化平均或各折平均計算；是否要求連續多輪達標。
