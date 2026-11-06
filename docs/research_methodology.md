# ADoRE-DoE 研究方法論

> 狀態：方法規格。尚無實作與實驗結果。標示為「開放問題」的項目是尚未決定的研究選擇。

## 1. 研究動機

見 README §2。核心問題：在實驗成本高昂時，如何以最少 run 數達成預定預測準確度，同時保留 RSM 的可解釋性。

## 2. 已確認的研究架構

1. 初始設計為 RSM-based（CCD、I-optimal 等）；Screening 為選用前置。
2. **Module 1**：以 GPR 為初始代理模型的 Active Learning，目標為全域預測改善；可獨立執行。
3. 多響應：共用實驗點、各響應獨立建模與評估。
4. Active Learning 結束後建立**最終 RSM**（保留方程式與統計可解釋性）。
5. **Module 2**：以 ML 預測 RSM 殘差；可獨立執行；不是 Module 1 的前置條件。
6. 四組 benchmark（A–D）。
7. GA 僅用於最終最佳化。

## 3. 記號

- 因子 $x\in\Omega\subset\mathbb{R}^k$（以 coded 單位表示，預設 $[-1,1]^k$ 或其約束子區域）。
- 響應 $m=1,\dots,M$，觀測 $y_{i}^{(m)}$， $i=1,\dots,n$，所有響應共用同一組 $x_i$。
- 多項式模型 $y = f(x)^\top\beta+\varepsilon$， $f(x)$ 為模型展開（常數、線性、交互作用、平方項）。
- 設計矩陣 $X\in\mathbb{R}^{n\times p}$，第 $i$ 列為 $f(x_i)^\top$。

## 4. I-optimality 規格

### 4.1 定義

對指定的回歸模型與設計區域 $\Omega$，最小化平均預測變異：

$$
I(\xi)=\frac{\int_\Omega \mathrm{Var}[\hat y(x)]\,w(x)\,dx}{\int_\Omega w(x)\,dx},\qquad
\mathrm{Var}[\hat y(x)]=\sigma^2 f(x)^\top (X^\top X)^{-1} f(x).
$$

等價地 $I \propto \mathrm{tr}\!\left[(X^\top X)^{-1}B\right]$，其中 moment matrix
$B=\int_\Omega f(x)f(x)^\top w(x)\,dx \big/ \int_\Omega w(x)\,dx$。

### 4.2 實作時必須明確記載的項目

| 項目 | 規格 |
|---|---|
| 模型矩陣 | 明列 $f(x)$ 的項（linear / interaction / quadratic），並用 coded 座標 |
| 設計區域 | 超立方體或帶線性 / 非線性約束的子區域 |
| 權重 $w(x)$ | 預設均勻；其他權重須明示 |
| 候選集 | 候選點的產生方式（網格、Sobol、LHS）與大小，須可由 seed 重現 |
| 數值近似 | $B\approx \frac1m\sum_j f(u_j)f(u_j)^\top$， $u_j$ 為區域內的積分點；須記錄 $m$ 與近似誤差檢查 |
| 秩與奇異性 | 檢查 $\mathrm{rank}(X)=p$ 與條件數；不可逆時回報而非靜默使用偽逆 |
| 約束區域 | 候選點與積分點都必須落在可行域內；可行域過小時回報 |
| 用途 | 同時用於 RSM Design、Design Augmentation、Sequential Design |

D-optimality（最大化 $\det(X^\top X)$）保留作比較。

### 4.3 與 GPR 預測不確定度的區別

| | 經典 I-optimality | GPR 後驗變異 |
|---|---|---|
| 模型 | 指定的線性（多項式）模型 | 由 kernel 與超參數決定的非參數先驗 |
| 變異公式 | $\sigma^2 f^\top(X^\top X)^{-1}f$ |  $k(x, x) - k_{\ast}^\top (K + \sigma_n^2 I)^{-1} k_{\ast}$ |
| 與 $y$ 的關係 | 僅依賴設計點 | 超參數固定時僅依賴設計點；若每輪重新估計超參數，則間接依賴觀測值 |
| 含意 | 多項式模型下的平均預測變異 | 代理模型的不確定度，受 kernel 假設影響 |

**兩者不可互換。** 以 GPR 變異平均化的準則，應稱為 integrated GPR predictive variance
（或 IMSE / IVR），**不得**稱為經典 I-optimality。只有在特定條件（例如線性 kernel 的貝氏線性回歸等價形式）
才存在對應關係，且須另行證明與測試。

## 5. Module 1：Active Learning

### 5.1 工作流程

1. 接收初始 RSM 實驗資料。
2. 訓練 surrogate（GPR，每個響應各一個）。
3. 產生候選實驗點。
4. 在固定評估集上評估全域預測品質與不確定度。
5. 以 acquisition 策略評估候選點。
6. 建議下一點或一批（batch）。
7. 接收新實驗結果並更新資料集。
8. 重新訓練。
9. 重新評估停止條件。
10. 重複直到停止條件成立或達 run 預算。

### 5.2 候選策略（待比較）

| 策略 | 定義 | 備註 |
|---|---|---|
| Maximum Predictive Variance | 選 $\arg\max_c \sigma^2(c)$ | baseline；傾向選邊界 / 外圍點，**不保證**全域改善 |
| Integrated Variance Reduction（IVR） | 選 $\arg\min_c \frac1m\sum_j \sigma^2_{n+1}(u_j\mid X\cup\{c\})$ | 超參數固定時，後驗變異不需 $y$ 即可預先計算（rank-1 更新） |
| I-optimal Augmentation | 在指定多項式模型下，對現有設計做 I-optimal 擴充 | 屬 `design`；用於「傳統式」擴充對照 |
| 其他 | 例如 worst-response、加權版本 | 待定 |

> 全域導向：策略比較的依據是評估集上的全域指標，而非單點的不確定度大小。

### 5.3 固定評估集

- 覆蓋預期設計區域（Sobol / LHS / 網格，符合約束），**與訓練資料獨立**，各 benchmark 群組**完全相同**，
  支援多響應，並由 seed 重現。
- 追蹤：Global RMSE、Global MAE、Global predictive R²、各響應指標，以及適用時的平均預測變異。
- **真值不可得時**（真實實驗）：評估點沒有 $y$，只能計算模型不確定度（平均預測變異）與以 CV 估計的預測誤差，
  **不得**編造評估點的真值。兩者必須分開報告，不得混稱「預測誤差」。
- **真值可得時**（以模擬器、解析函數或預先收集的大型資料池做 benchmark）：才可計算評估集上的真實 RMSE / MAE / R²。

## 6. 多響應

### 6.1 Acquisition 聚合

不同響應單位與尺度不同，**原始變異不可直接相加**。必須明確正規化，預設提案：

$$
\tilde\sigma_m^2(x)=\frac{\sigma_m^2(x)}{\widehat{\mathrm{Var}}(y^{(m)})}
$$

即以該響應資料變異數（或由評估集上的先驗變異）標準化，之後再聚合：

| 聚合 | 公式 | 特性 |
|---|---|---|
| 加權平均 | $A(x)=\sum_m w_m\tilde\sigma_m^2(x),\ \sum w_m=1$ | 預設， $w_m$ 須明示（預設等權） |
| 最差響應 | $A(x)=\max_m \tilde\sigma_m^2(x)$ | 偏保守 |
| 積分變異降低 | 對聚合量在評估集上積分 | 與 IVR 搭配 |

開放問題：正規化分母用資料變異數、先驗變異或評估集變異？ 待 Phase 3 以實驗決定。

### 6.2 停止條件

停止須**同時**滿足：

1. 每個響應的預測 $R^2_m \ge \tau_m$（可各自設定），**且**
2. 聚合 $R^2$（平均或加權平均） $\ge \tau_{\text{agg}}$。

範例：三個響應各 $\ge 0.90$，且平均 $\ge 0.95$。

另外支援：最大 run 預算、batch size、cross-validation 策略（K-fold / LOO / 重複 CV）。

規則：

- 訓練 $R^2$ **不得**是唯一停止依據。
- 預測 $R^2 = 1-\mathrm{SSE}/\mathrm{SST}$，SSE 取自 held-out（OOF）預測；**負值必須保留**，不得截斷為 0。
- 無法計算時（例如響應為常數使 SST = 0、樣本數不足以做 CV）：回報原因，該響應視為未達標。
- 預算用盡但未達標：停止原因記為 `budget_exhausted`，明確標示「目標未達成」。

開放問題：是否要求連續 $k$ 輪皆達標才停止，以降低 CV 估計的雜訊？

## 7. 最終 RSM 建模

一階 / 二階多項式、stepwise、AICc / BIC、hierarchy 約束、ANOVA、Lack-of-Fit（需重複點或純誤差估計）、
模型統計量與殘差診斷。保留明確方程式。

開放問題：Lack-of-Fit 需要純誤差；序貫選出的點可能缺少重複點，須在 Phase 2/3 決定是否強制加入重複 / 中心點。

## 8. Module 2：殘差修正

$e_i=y_i-\hat y_{\mathrm{RSM},i}$， $\ \hat y_{\mathrm{final}}(x)=\hat y_{\mathrm{RSM}}(x)+\hat e_{\mathrm{ML}}(x)$（每個響應各自一個）。

### 8.1 OOF 流程（避免洩漏）

1. 將資料分成 $K$ 折（小樣本用 LOO 或重複 K-fold）。
2. 對每折：只用其餘折擬合 RSM， **包含模型選擇**（stepwise / AICc 也要在折內重做，否則選擇步驟洩漏），
   再預測該折 → 得到 OOF 預測 $\hat y^{(-i)}$。
3. 殘差目標 $e_i=y_i-\hat y^{(-i)}_{\mathrm{RSM},i}$。
4. 以 $(x_i,e_i)$ 擬合殘差模型。
5. 最終 RSM 用全部資料擬合；hybrid 預測 = 全資料 RSM + 殘差模型。
6. 在**獨立驗證資料**上評估 hybrid，並與純 RSM 比較($\Delta\text{RMSE}$、 $\Delta R^2$)。

### 8.2 注意事項

- OOF 殘差通常比全資料 RSM 的殘差大（每折模型用的資料較少），殘差分布與最終使用情境略有偏差；須在結果中說明。
- 小樣本 DOE 下 CV 估計方差大，須文件化限制；不預設 ANN 最佳，並應包含「不修正」作為對照，
  殘差模型若無泛化增益應允許退回純 RSM。
- 評估看泛化表現，**不看訓練殘差下降**。

## 9. 四組 Benchmark

| 組別 | 設計 | 最終模型 |
|---|---|---|
| A | Traditional DOE | RSM |
| B | Adaptive DOE | RSM |
| C | Traditional DOE | RSM + ML Residual Correction |
| D | Adaptive DOE | RSM + ML Residual Correction |

### 9.1 公平性規則

相同初始實驗條件與初始 run 數 $n_0$、相同獨立驗證集（以及相同評估集）、相同 run 預算、
顯式 seed、多 seed 重複並回報平均與標準差（亦建議回報中位數 / IQR）。

### 9.2 比較方式

- **固定預算**：例如 15 / 20 / 25 / 30 runs，報告 RMSE、MAE、R²。
- **目標準確度**：達標所需 run 數。未達標者以 **右設限（censored** 處理：回報「在 $k/N$ 個 seed 中未達標」，
  不得把預算值當成達標 run 數直接平均而不加標註。

### 9.3 開放問題：Traditional DOE 的 baseline 定義

| 選項 | 說明 | 利弊 |
|---|---|---|
| (a) 一次性設計 | 對每個預算 $N$ 直接產生 $N$-run 的 CCD / I-optimal 設計 | 最貼近傳統 DOE；但不同 $N$ 的設計不巢狀，且不含共同的初始 $n_0$ 點 |
| (b) 初始設計 + 一次性 I-optimal 擴充 | 以同一 $n_0$ 起點擴充至 $N$（非回應導向） | 與 B/D 共享初始點，更能隔離「adaptive」的效果 |

建議：以 (a) 為主、(b) 為 ablation 一併報告，Phase 6 前定案。

### 9.4 關於「adaptive」的誠實說明

在固定超參數下，GPR 後驗變異與觀測值 $y$ 無關，acquisition 是否真正「回應導向」取決於是否每輪重新估計超參數
（kernel length-scale 等）。報告中必須說明所用設定，並考慮加入「固定超參數」與「逐輪重估」兩種版本比較。

## 10. 評估指標

| 指標 | 定義 |
|---|---|
| RMSE | $\sqrt{\frac1m\sum(y-\hat y)^2}$ |
| MAE | $\frac1m\sum\lvert y-\hat y\rvert$ |
| 預測 $R^2$ | $1-\mathrm{SSE}/\mathrm{SST}$（held-out；可為負） |
| 平均預測變異 | 評估集上 $\overline{\sigma^2}$（模型不確定度，不是誤差） |
| 達標 run 數 | 首次滿足全部停止條件的 run 數，或「未達成」 |

## 11. 已知方法學限制

1. **CV 偏誤**：序貫選點使資料分布不均（集中於高變異或邊界），以這些資料做 CV 估計的 $R^2$ 可能偏離對均勻評估集的真實 $R^2$。
2. **代理模型不匹配**：Module 1 以 GPR 的指標判斷停止，但最終模型是 RSM（多項式）；GPR 達標不代表 RSM 達標。
   Benchmark 階段必須同時報告最終模型（RSM、hybrid）的驗證表現。
3. **小樣本**：CV 與殘差模型估計方差大；停止條件的 $R^2$ 雜訊高。
4. **評估集真值**：真實實驗無法在大量評估點取得真值，需依賴模擬器 / 資料池或 CV 代理。
5. **多重比較**：多策略 × 多預算 × 多響應的比較需注意結論的穩健性。
6. **外插**：RSM 與 hybrid 在設計區域外不可信；最佳化階段需外插警告。
7. **尚無實證**：以上皆為方法規格，尚未有數值驗證。
