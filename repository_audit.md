# 舊 Repository 審查報告（DoE-ML-GA / YourPkg）

審查對象：`https://github.com/Kelvin-Hua05/DoE-ML-GA`（main 分支，6 commits，公開頁面）。
審查方式：讀取 repo 首頁與 README。**限制**：`ref_paper/` 目錄內容因網站禁止自動存取而**未能檢視**；
也無法執行 `git status`、測試或檢視 commit 內容。

## 1. 現況

| 項目 | 結果 |
|---|---|
| 檔案 | 僅 `README.md` 與 `ref_paper/` |
| `src/`、`tests/`、`examples/`、`docs/`、`pyproject.toml`、`LICENSE` | 首頁檔案列表中**不存在**（README 中的連結指向這些路徑，但無對應檔案） |
| 套件名稱 | `YourPkg`（佔位）；badge 與連結含 `YOUR_NAME` 佔位字串 |
| 實作 | **沒有任何原始碼**，因此沒有現有 API 需要保留，也沒有測試可執行 |

## 2. 功能審查表

| Feature | Existing Status | Existing Location | Required Action |
|---|---|---|---|
| Screening | Missing（僅 README 規劃） | — | Add（Phase 1） |
| RSM Design | Missing | — | Add（Phase 2） |
| I-optimal | Missing（舊規劃以 D-optimal 為主） | — | Add，整合入 RSM Design / Augmentation（Phase 2） |
| Active Learning | Missing（舊規劃完全沒有） | — | Add（Phase 3，最高優先） |
| RSM Modeling | Missing | — | Add（Phase 4） |
| Residual Correction | Missing（舊規劃有 Hybrid RSM+GP/RF 與 WLS/轉換） | — | Add，改為 OOF 設計（Phase 5） |
| Benchmark | Missing（舊規劃完全沒有） | — | Add（Phase 6，最高優先） |
| Optimization | Missing | — | Add（Phase 7） |
| Sensitivity | Missing | — | Add（Phase 7） |
| Report 輸出 | Missing | — | Defer |

## 3. 舊規劃與新架構的對應

| 舊 README 結構 | 新位置 | 說明 |
|---|---|---|
| `design/`（PB、CCD、D-optimal、augmentation） | `screening/`（PB、FF）＋ `design/`（CCD、I-/D-optimal、augmentation） | Screening 拆出為選用前置 |
| `analysis/` | `screening/analysis.py`、`rsm/anova.py` | 依用途分到兩處 |
| `rsm/` | `rsm/` | 保留 |
| `diagnostics/` | `rsm/diagnostics.py` | 診斷屬於 RSM 建模 |
| `hybrid/` | `residual/hybrid.py` | 改名並加入 OOF 規範 |
| — | `active_learning/`、`benchmark/` | **新增**（舊規劃沒有） |
| `optimize/`、`sensitivity/` | 同名 | 保留，延後 |
| `report/` | — | 延後 |
| `pipeline.py` | `pipeline.py` | 延後 |

## 4. 舊 README 中**未**沿用的內容與原因

- `pip install yourpkg`、PyPI / CI / codecov / Docs badge：套件未發佈，連結為佔位，屬於不實宣稱。
- 「快速開始」程式碼（`yp.design.plackett_burman(...)`、`yp.Pipeline.from_yaml(...)` 等）：沒有對應實作，屬於虛構 API。
- 「商業軟體（JMP / Minitab / Design-Expert）比對」：沒有任何驗證證據。
- 舊版 v0.1–v1.0 roadmap：與已確認的優先順序不符，由 `development_roadmap.md` 取代。
- 「殘差修正」中的轉換 / WLS：屬 RSM 假設補救，與 ML 殘差修正是不同概念，列入延後項目。

## 5. 遷移風險

| 風險 | 說明 |
|---|---|
| PyPI 名稱衝突 | `adore` 已被他人佔用；見 ADR-007。發佈名稱暫定 `adore-doe`，尚未註冊 |
| import 名稱衝突 | 若同環境裝有 PyPI 的 `adore`，頂層模組名稱會衝突 |
| 舊 repo 內容 | 只有 README 與 `ref_paper/`；若需保留歷史，需決定是沿用舊 repo 還是新開（你已決定新開） |
| `ref_paper/` | 內容未能檢視，請自行搬移；若含受版權保護的論文 PDF，請確認是否適合放入公開 repo |
| 授權 | 舊 README 聲稱 MIT，但沒有看到 `LICENSE` 檔；新 repo 尚未加入授權，需由擁有者決定 |
