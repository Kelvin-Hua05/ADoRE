# Phase 2：RSM Design 與 I-optimal

範圍：CCD（ADR-010）、I-optimal（主要）、D-optimal（比較）、augmentation、候選集、線性約束區域、秩 / 條件數檢查、FDS 圖，以及隨之完成的 seed 管理。
**不做**：Box-Behnken（ADR-008 已移除）、fractional CCD 與 α>1 的 CCD（ADR-010）、非線性約束（ADR-001）。

## 1. API

```python
import adore_doe as ad
from adore_doe import design

factors = [ad.Factor("Width", 10, 50), ad.Factor("Spacing", 5, 30), ad.Factor("Thickness", 20, 80)]

d_ccd = design.ccd(factors, center_points=1)                      # 2^k + 2k + 1 = 15 runs
d_i   = design.i_optimal(factors, "quadratic", runs=15, seed=1)   # runs 無預設值
d_d   = design.d_optimal(factors, "quadratic", runs=15, seed=1)

# 線性約束（實際單位）、非均勻權重、強制點
d_c = design.i_optimal(factors, "quadratic", runs=14, seed=1,
                       constraints=[({"Width": 1, "Spacing": 1}, "<=", 60)])

aug = design.augment(d_i, 3, criterion="I", seed=2)               # 只「建議」，不會執行或寫入任何模擬
aug.to_csv("next_runs.csv"); aug.to_dataset(d_i)                  # 之後才把建議點加入資料集（response 為 NaN）

design.plot_fds({"CCD": design.fds(d_ccd, seed=1), "I-opt": design.fds(d_i, seed=1)})
```

所有設計回傳 `ExperimentDataset`；`info` 保存 `model_spec`、準則值、秩、條件數、積分點設定與收斂檢查、候選集資訊、區域、seed，
供後續 RSM 建模階段直接使用。

## 2. 準則定義（σ² = 1）

* **I-criterion**：`I(X) = tr[(X'X)⁻¹ B]`，`B = ∫ f f' w dx / ∫ w dx`（越小越好）= 設計區域上的平均預測變異。
* **D-criterion**：`log det(X'X)`，由 QR 的 `2 Σ log|R_ii|` 計算（不直接算行列式）；`D_efficiency = det(X'X)^(1/p) / n`。
* 兩者都會在回報前做秩檢查；奇異設計**報錯**（`RankDeficientError`），**不會**靜默使用偽逆。

增量資訊量（Phase 3 的 I-optimal augmentation 也使用同一函式 `candidate_scores`）：
`det(A + ff') = det(A)(1 + f'A⁻¹f)`（行列式引理）；`tr[(A+ff')⁻¹B] = tr[A⁻¹B] − f'A⁻¹BA⁻¹f / (1 + f'A⁻¹f)`（Sherman–Morrison）。兩式皆有測試以「重新計算準則」逐點驗證。

## 3. research_methodology §4.2 規格逐項對照

| 項目 | 實作 | 記錄位置 / 參數 |
|---|---|---|
| 模型矩陣 | `ModelSpec(kind, k)`：`linear`（1, xᵢ）、`interaction`（＋xᵢxⱼ）、`quadratic`（＋xᵢ²，p=(k+1)(k+2)/2）；一律 coded 座標；`terms` 的順序＝X 的欄順序 | `info["model_spec"]`、`ModelSpec.names()` |
| 設計區域 | 立方體 [−1,1]ᵏ，可加**線性**約束 `Az ≤ b`（實際單位輸入，內部轉 coded）；非線性約束不支援 | `constraints=` / `region=`；`info["region"]` |
| 權重 w(x) | 預設均勻；其他以 `weight=callable(Z)->非負權重` 明示，並記錄為 `custom` | `info["weight"]` |
| 候選集 | `generate_candidates`：`grid`（levels^k 含頂點與中點）、`sobol`、`lhs`、`uniform`、`user`、`auto`；可由 seed 重現；回報大小與接受率；可排除既有點 | `info["candidates"]` |
| 數值近似 | `B ≈ Σ wⱼ f(uⱼ)f(uⱼ)' / Σ wⱼ`，uⱼ 為區域內的 scrambled-Sobol 點；**m 可設定**（`m=`）。①`B` 的相對 Frobenius 變化隨 m 加倍檢查；②**對最終設計**重算 I-criterion（m, 2m, 4m…）檢查**準則值**的相對變化（預設容許 1%，`tol=`，ADR-001），未收斂會警告 | `info["moment"]`、`info["moment_check"]`（m、各次相對變化、converged） |
| 秩與奇異性 | `check_design`（SVD）：rank、條件數；`rank<p` 報錯並說明（runs<p 或候選集不足）；cond(X)>1e8 警告；runs==p 警告「無誤差 df」 | `info["rank"]`、`info["cond_X"]` |
| 約束區域 | 候選點與積分點都以 rejection sampling 限制在可行域；可行域為空或接受率低於 `min_acceptance`（預設 1e-3）時 `InfeasibleRegionError`；記錄接受率 | `info["moment"]["acceptance_rate"]`、`info["candidates"]["acceptance_rate"]` |
| 用途 | 同一組機制（模型＋區域＋候選集＋積分域）用於初始設計、`augment`，並供 Phase 3 的序貫設計呼叫 | — |

> 關於「數值近似」的說明：在**無約束立方體 + 均勻權重**下，Sobol 點對低階多項式 moment 幾乎是精確的（相對變化為 0），所以檢查會立即通過；有約束或自訂權重時才有真正的近似誤差，檢查才有意義。解析解 `exact_uniform_moment` 僅供對照測試。

## 4. 設計方法

**CCD（ADR-010）**：full factorial 2ᵏ（Yates 順序，第一個因子變動最快）＋軸點（±1，face-centered）＋中心點（預設 **1** 個）。run 數 `2ᵏ + 2k + n_center`：k=2：9，k=3：15，k=4：25，k=5：43（會警告超出 10–30 run 預算），上限 k=8。所有點都在因子上下限內。**只有 1 個中心點時沒有純誤差，無法做 lack-of-fit 檢定**（`info["lof_note"]`）；需要 LOF 請用 `center_points≥2` 或另外加重複點。k=2 的 face-centered CCD 就是 3² 設計（有測試）。

**I-/D-optimal**：候選集上的 candidate-exchange（第 1 個起點為貪婪，其餘為隨機起點），一次換一點、直到沒有改善。`runs` 為總 run 數（含 `center_points` 與 `include` 強制點）；預設不允許重複點（確定性模擬重複無資訊），`allow_replicates=True` 才允許。
**這是啟發式搜尋，不保證全域最佳**：實測（8 個 seed）：k=3、15 runs 的 **I-optimal** 20 個起點中只有 1–2 個找到最佳解（同條件的 D-optimal 為 12–19 個）；k=4、25 runs 的 I-optimal 只有 1 個。I-optimal 的準則曲面局部最佳較多，因此預設 `n_starts=20`，並把「有幾個起點達到最佳值」記錄在 `info["starts_reaching_best"]`。**這個數字是 1 時，表示最佳解只被一個起點找到，不能排除還有更好的解；因子數 ≥ 4 時建議把 `n_starts` 提高到 50 以上，並用不同 seed 重跑確認準則值穩定。**

**Augmentation**：保留既有 run，從尚未使用的候選點選出 `n_add` 個最能改善準則的點（驗證資料不算設計的一部分）。`n_add=1` 時與「逐一嘗試所有候選點」的結果完全一致（有測試）。既有設計不可識別時警告，並以最終設計的秩作為成敗依據。

### 4.1 實測觀察（不是證明）

* **k=3、15 runs**：I-optimal（5 水準格點候選集）找到的點集與 face-centered CCD **完全相同**（seed 1、2、3 皆是），因此兩者準則值與 FDS 曲線重疊。這不是錯誤——在這個條件下 CCD 本身就是候選集上 I 準則的最佳設計。D-optimal 則不同（I=0.507 對 0.368，log det 較大）。
* **k=4、25 runs**：I-optimal（`n_starts=40`）I=0.3016，優於 CCD 的 0.3376。
* 兩者都只是這些設定下的實測；I-optimal 的優勢會隨 k、run 數、候選集與約束而變，**不應推廣為一般結論**。

## 5. 與 GPR 預測變異的區別（不可混用）

本階段的 I-criterion 是**指定多項式模型**的經典設計變異 `σ² f'(X'X)⁻¹f`，只依賴設計點。GPR 後驗變異由 kernel 與超參數決定，兩者不可互換。以 GPR 變異平均化的準則應稱為 integrated GPR predictive variance（IVR），**不是** I-optimality（見 research_methodology §4.3）。`fds` 繪製的是多項式模型的變異，不是 GPR 的。

## 6. Seed 管理（`core/seed.py`）

* 所有隨機程序都接受 `seed`，**不使用 NumPy 全域亂數**；`seed=None` 會產生一個種子並記錄（`info["seed"]`），事後可重現。
* 一個 root seed 以名稱分出獨立的子串流（`candidates`、`moment`、`starts`、`randomize`、`fds`…），改變其中一個元件不會影響其他元件；實際使用的子 seed 記錄在 `info["seeds"]`。
* 子 seed 由 SHA-256 導出（不是 Python 的 `hash`，後者每個行程不同），跨行程、跨機器相同（有測試，並釘住數值）。
* 接受的型別：非負整數或 None（bool、float、負數會報錯）。

## 7. 驗證範圍與限制（誠實聲明）

已驗證（pytest）：手算的小例子（1 因子線性 I=2/3、det=4；1 因子二次 {−1,0,1} 的 I=4/5、det=4）；準則的增量公式；**exchange 演算法與窮舉搜尋的最佳解一致**（4 組小問題 × 2 種準則）；CCD 點數與結構；約束區域內所有點可行；同 seed 可重現；奇異設計報錯。
**未做**：與 R 的 `AlgDesign` / `rsm` 對照（沙箱無 R、無網路），因此**不宣稱**數值與它們一致；大問題上的全域最佳性沒有證明；非線性約束、fractional CCD、α>1、BBD 不支援。
