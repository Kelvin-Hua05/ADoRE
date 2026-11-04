<div align="center">

# 🧪 ADoRE

**Adaptive Design of Experiments with Residual Enhancement — 從 Screening、自適應實驗設計、RSM 建模、ML 殘差修正到最佳化與敏感度分析的一站式 DoE 工具包**

*An end-to-end adaptive DoE toolkit: Screening → RSM Design → Active Learning → RSM Modeling → ML Residual Correction → Benchmark → Optimization → Sensitivity Analysis*

[![PyPI](https://img.shields.io/pypi/v/adore.svg)](https://pypi.org/project/adore/)
[![Python](https://img.shields.io/pypi/pyversions/adore.svg)](https://pypi.org/project/adore/)
[![CI](https://github.com/Kelvin-Hua05/ADoRE/actions/workflows/ci.yml/badge.svg)](https://github.com/Kelvin-Hua05/ADoRE/actions)
[![codecov](https://codecov.io/gh/Kelvin-Hua05/ADoRE/branch/main/graph/badge.svg)](https://codecov.io/gh/Kelvin-Hua05/ADoRE)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://kelvin-hua05.github.io/ADoRE)

[快速開始](#-快速開始) · [功能特色](#-功能特色) · [文件](#-文件與索引) · [Roadmap](#-roadmap)

</div>
---

> ⚠️ **狀態：Pre-alpha（文件與架構階段）**
> 目前**沒有任何研究演算法被實作**。repo 內只有套件骨架（`import adore`、`adore.__version__`）、架構文件與開發規劃。
> 下文所有功能皆標示實作狀態，**未標示 ✅ 者都不能使用**；PyPI、CI、Codecov、Docs 徽章在對應服務建立前可能顯示為未知或失敗，屬預期狀態。

## 📖 簡介

ADoRE 把「實驗點怎麼選」與「模型怎麼補」整合成單一 Python 工具包，目標是用**最少的昂貴實驗**達到預先設定的預測準確度，同時保留傳統 RSM 的可解釋性（明確方程式、ANOVA、Lack-of-Fit）：

1. **Screening**：Plackett-Burman / DSD 篩選關鍵因子（選用）
2. **RSM Design**：CCD、Box-Behnken、I-optimal（含 D-optimal 對照）、Augmentation
3. **Active Learning（Module 1）**：以 GPR 代理模型序貫建議下一個實驗點，目標是全域預測改善
4. **RSM 建模與診斷**：二階多項式、Stepwise、AICc / BIC、ANOVA、Lack-of-Fit、殘差診斷、轉換 / WLS
5. **ML 殘差修正（Module 2）**：`ŷ_final = ŷ_RSM + ê_ML`，以 out-of-fold 殘差避免洩漏
6. **Benchmark**：四組對照量化「Adaptive DOE」與「Residual Enhancement」各自與合併的效益
7. **最佳化**：Desirability、SQP、GA、NSGA-II
8. **敏感度分析**：Perturbation、Tornado、Sobol、Morris、Monte Carlo 穩健性分析

### 為什麼做這個？

傳統 DOE 一次決定整組實驗點，不會根據已得資料調整；傳統 RSM 的多項式模型在曲率複雜時，殘差中常含有結構性訊號。ADoRE 同時處理這兩件事：實驗點「怎麼選」（Adaptive DOE），以及模型「怎麼補」（Residual Enhancement），並在相同條件下做公平對照。

## ✨ 功能特色

| 階段 | 功能 | 狀態 |
|------|------|:----:|
| 套件骨架 | `import adore`、`adore.__version__` | ✅ |
| Core | 因子 / 響應 / 資料集結構、顯式 seed 管理 | 📋 |
| Screening | Plackett-Burman、Fractional Factorial、DSD、主效應 / Pareto / Half-normal | 📋 |
| RSM 設計 | CCD、Box-Behnken、I-optimal（主要）、D-optimal（對照）、Augmentation、FDS plot | 📋 |
| **Active Learning** | GPR 代理、Max Predictive Variance、IVR、I-optimal Augmentation、批次建議、多響應停止條件 | 📋 |
| RSM 建模 | 一階 / 二階模型、Stepwise、AICc / BIC、Hierarchy 約束、ANOVA、Lack-of-Fit | 📋 |
| 殘差診斷 | Normal plot、Residual vs Fitted / Order / Factor、Cook's D、Leverage、Box-Cox | 📋 |
| 統計修正 | 資料轉換、WLS | 📋 |
| **ML 殘差修正** | GPR / ANN / Random Forest / SVR、OOF 殘差、獨立驗證集評估 | 📋 |
| **Benchmark** | 四組比較（固定 run 預算 / 目標準確度）、多種子重複 | 📋 |
| 最佳化 | Desirability、SQP、GA、NSGA-II、外插警告 | 📋 |
| 敏感度分析 | Perturbation、Tornado、Sobol、Morris、Monte Carlo | 📋 |
| 報告 | HTML / PDF / Excel 自動輸出 | 📋 |
| Pipeline | YAML 設定一次串接全流程 | 📋 |

✅ 已完成　🚧 開發中　📋 規劃中

## 🔄 整體流程

```
[選用] Screening（PB / DSD）
        │
        ▼
RSM Design（CCD / I-optimal / …）──► 初始實驗資料
        │
        ▼
Module 1：Active Learning（GPR 代理模型，可獨立執行）
   訓練 → 候選點 → 全域評估 → acquisition → 建議下一點 / 批次
   → 取得新實驗結果 → 更新 → 檢查停止條件 → 重複
        │  累積實驗資料
        ▼
最終 RSM 建模（多項式、Stepwise、AICc / BIC、ANOVA、LOF、診斷）
        │
        ▼
Module 2：ML 殘差修正（可在給定 RSM 與資料時獨立執行）
        │
        ▼
Benchmark（四組比較）   最佳化（Desirability / SQP / GA / NSGA-II）   敏感度分析
```

詳細圖示見 [`docs/architecture.md`](docs/architecture.md)。

## 📦 安裝

目前**尚未發佈到 PyPI**，僅支援從原始碼安裝（**需求**：Python ≥ 3.10）：

```bash
git clone https://github.com/Kelvin-Hua05/ADoRE.git
cd ADoRE
pip install -e ".[dev]"
pytest
```

發佈後預計提供（📋 尚不可用）：

```bash
pip install adore            # 核心
pip install "adore[all]"     # 含最佳化 / 敏感度分析 / 報告輸出
```

## 🚀 快速開始

**目前唯一可用的 API：**

```python
import adore
print(adore.__version__)
```

**以下為概念性流程（pseudocode），所有呼叫都尚未實作，僅用於說明預期用法：**

```python
import adore as ad

# 1. 定義因子與響應
factors = [
    ad.Factor("Width",   low=0.1, high=0.5, unit="um"),
    ad.Factor("Spacing", low=0.1, high=0.5, unit="um"),
    ad.Factor("Height",  low=0.2, high=1.0, unit="um"),
]
responses = [ad.Response("C"), ad.Response("R")]

# 2. （選用）Screening
screen = ad.design.plackett_burman(factors, runs=12, center_points=3)

# 3. RSM 初始設計（I-optimal）
initial = ad.design.rsm_design(factors, method="i_optimal", n_runs=15, seed=0)

# 4. Module 1：Active Learning 迴圈
al = ad.active_learning.Loop(
    surrogate="gpr",
    acquisition="ivr",
    batch_size=2,
    stopping=ad.StoppingRule(min_r2_each=0.90, min_r2_aggregate=0.95, max_runs=40),
    seed=0,
)
dataset = initial
while not al.stopped(dataset):
    next_points = al.recommend(dataset)
    dataset = dataset.add(run_experiment(next_points))   # 使用者自行執行實驗

# 5. 最終 RSM 建模與診斷
rsm_model = ad.rsm.fit(dataset, selection="aicc")
rsm_model.anova()
rsm_model.diagnose()

# 6. Module 2：ML 殘差修正（OOF）
hybrid = ad.residual.fit(rsm_model, dataset, learner="gpr", cv="oof")

# 7. 最佳化與敏感度分析
opt = ad.optimize.run(hybrid, goal="minimize", method="ga")
ad.sensitivity.sobol(hybrid).plot()
```

預計提供的 Pipeline 用法（📋）：

```python
pipe = ad.Pipeline.from_yaml("config.yaml")
pipe.run()
pipe.report("report.html")
```

## 🧭 核心模組說明

### Module 1 — Active Learning（最高優先）

- 輸入初始 RSM 實驗資料 → 以 **GPR** 為初始代理模型 → 建議下一個實驗點或一批實驗點。
- 目標是 **Global Prediction Improvement**，**不是**單純挑預測不確定度最高的點。
- 預計研究的策略：Maximum Predictive Variance、Integrated Predictive Variance Reduction、I-optimal Augmentation，以及其他以全域預測為導向的 acquisition。
- **不依賴**最終 RSM 建模模組或 Module 2；**GA 不屬於** Active Learning。

### RSM Design 與 I-optimal 整合

- 初始設計必須是 RSM-based：CCD、Box-Behnken、I-optimal，必要時其他 optimal design。
- Screening 為選用前置階段；已知關鍵因子時可直接進入 RSM Design。
- **I-optimal 不是獨立開發階段**，而是整合在 RSM Design、Design Augmentation 與 Sequential Design 中；I-optimality 為主要準則，D-optimality 保留作比較。
- 經典線性模型的 I-optimality 與 GPR 預測不確定度**不可視為數學等價**（見 [`docs/research_methodology.md`](docs/research_methodology.md) §4）。

### 最終 RSM 建模

一階 / 二階多項式、Stepwise、AICc / BIC、Hierarchy 約束、ANOVA、Lack-of-Fit、模型統計量、殘差診斷與轉換 / WLS。最終模型必須保有**明確的數學方程式**與統計可解釋性。

### Module 2 — ML 殘差修正

對每個響應：`e_i = y_i − ŷ_RSM,i`，混合預測 `ŷ_final(x) = ŷ_RSM(x) + ê_ML(x)`。

- 候選模型：GPR、ANN、Random Forest、SVR 等；**不預設 ANN 最佳**。
- 使用 cross-validation 時，殘差目標必須由 **out-of-fold（OOF）RSM 預測**建立，避免 in-sample 洩漏。
- 保留獨立驗證 / 確認資料；以**泛化表現**而非訓練殘差下降評估。
- 統計層面的轉換 / WLS 屬於 RSM 建模，與 ML 殘差修正分開處理。

### 多響應停止條件

所有響應共用同一組實驗點，但**各自建模、各自評估**。停止需**同時**滿足：

- 每個響應的個別最小預測 R²（例：各 ≥ 0.90），**且**
- 聚合預測 R²（平均或加權平均，例：≥ 0.95），

或達到最大 run 預算（此時必須標示「目標未達成」）。訓練 R² 不得作為唯一停止依據；負的 R² 必須保留；無法計算時回報原因並視為未達標。

### 最佳化與敏感度分析

GA 保留給**最終製程 / 設計參數最佳化**（Desirability、SQP、GA、NSGA-II），作用於已驗證的最終預測模型。敏感度分析（Perturbation、Tornado、Sobol、Morris、Monte Carlo）同屬後期項目。

## 📊 四組 Benchmark

| 組別 | 實驗設計 | 最終模型 |
|:----:|----------|----------|
| A | Traditional DOE | RSM |
| B | Adaptive DOE | RSM |
| C | Traditional DOE | RSM + ML Residual Correction |
| D | Adaptive DOE | RSM + ML Residual Correction |

兩種比較方式：

- **固定 run 預算**（如 15 / 20 / 25 / 30 runs）：RMSE、MAE、R²。
- **目標準確度**：達標所需 run 數；未達標必須明確回報。

公平性要求：相同初始條件與初始 run 數、相同獨立驗證集、相同預算、可重現亂數種子、多種子重複並回報平均與標準差。

## 🗂 專案結構

目前實際存在：

```
ADoRE/
├── src/adore/__init__.py
├── tests/test_package.py
├── docs/
├── pyproject.toml
└── README.md
```

目標架構（**尚未建立**，各目錄於對應開發階段才建立，不預先建立空模組）：

```
src/adore/
├── core/              # Factor / Response / Dataset、seed 管理
├── screening/         # PB、DSD、主效應分析
├── design/            # CCD、I-/D-optimal、augmentation
├── active_learning/   # GPR 代理、acquisition、停止條件
├── rsm/               # 多項式模型、Stepwise、ANOVA、LOF、轉換、WLS
├── diagnostics/       # 殘差圖、Leverage、Cook's D、Box-Cox
├── residual/          # ML 殘差修正（OOF）
├── benchmark/         # 四組比較
├── optimize/          # Desirability、SQP、GA、NSGA-II
├── sensitivity/       # Perturbation、Tornado、Sobol、Morris、Monte Carlo
├── report/            # HTML / PDF / Excel 輸出
└── pipeline.py        # 全流程串接
```

模組職責與依賴規則見 [`docs/architecture.md`](docs/architecture.md)。

## ✅ 驗證與可重現性

- 目前沒有任何統計或演算法結果可供驗證，**不宣稱**與任何商業軟體相容，也不宣稱「已完整驗證」。
- 規劃的驗證方式：以教科書公開範例與 R 參考實作（`rsm`、`AlgDesign`、`FrF2` 等）做數值回歸測試；通過後才會在文件中記載對照結果。
- 預計參考教科書：Montgomery, *Design and Analysis of Experiments*；Myers et al., *Response Surface Methodology*。
- 所有隨機程序（候選點、評估集、初始設計、CV 切分）都必須接受顯式 seed。

## 🗺 Roadmap

> 順序對應 [`docs/development_roadmap.md`](docs/development_roadmap.md)；各階段的測試與驗收標準以該文件為準。完成一項就把 `[ ]` 改為 `[x]`。

- [x] **v0.0** 套件骨架（`import adore`、`adore.__version__`、smoke test）
- [ ] **v0.1** Core 資料結構 + Screening（Plackett-Burman / DSD）+ 主效應分析
  - [ ] Factor / Response / Dataset
  - [ ] 顯式 seed 管理
  - [ ] Plackett-Burman、DSD
  - [ ] 主效應 / Pareto / Half-normal
- [ ] **v0.2** RSM Design + I-optimal 整合
  - [ ] CCD、Box-Behnken
  - [ ] I-optimal（主要）與 D-optimal（對照）
  - [ ] Design Augmentation、FDS plot
- [ ] **v0.3** Module 1：Active Learning ⭐ 最高優先
  - [ ] GPR 代理模型
  - [ ] Acquisition：Max Variance、IVR、I-optimal Augmentation
  - [ ] 批次建議
  - [ ] 多響應停止條件
- [ ] **v0.4** 最終 RSM 建模與診斷
  - [ ] 二階模型、Stepwise、AICc / BIC、Hierarchy 約束
  - [ ] ANOVA、Lack-of-Fit
  - [ ] 殘差診斷全套 + 轉換 / WLS
- [ ] **v0.5** Module 2：ML 殘差修正
  - [ ] OOF 殘差建立
  - [ ] GPR / ANN / RF / SVR 候選模型
  - [ ] 獨立驗證集評估
- [ ] **v0.6** 四組研究 Benchmark ⭐ 最高優先
  - [ ] 固定 run 預算比較
  - [ ] 目標準確度比較
  - [ ] 多種子重複與統計彙整
- [ ] **v0.7** 最佳化與敏感度分析
  - [ ] Desirability、SQP、GA、NSGA-II
  - [ ] Perturbation、Tornado、Sobol、Morris、Monte Carlo
- [ ] **v1.0** Pipeline、報告輸出、穩定 API、完整文件
  - [ ] YAML Pipeline
  - [ ] HTML / PDF / Excel 報告
  - [ ] 以參考實作完成數值回歸驗證
  - [ ] 發佈 PyPI、啟用 CI / Codecov / Docs

詳見 [Projects](https://github.com/Kelvin-Hua05/ADoRE/projects) 與 [Issues](https://github.com/Kelvin-Hua05/ADoRE/issues)。

## 📚 文件與索引

| 文件 | 內容 |
|------|------|
| [`docs/architecture.md`](docs/architecture.md) | 架構圖、模組職責、資料流、介面草案 |
| [`docs/research_methodology.md`](docs/research_methodology.md) | 數學目標、策略、停止條件、benchmark 方法 |
| [`docs/development_roadmap.md`](docs/development_roadmap.md) | 階段、依賴、測試、驗收標準 |
| [`docs/algorithm_decisions.md`](docs/algorithm_decisions.md) | ADR 形式的技術決策 |
| [`docs/repository_audit.md`](docs/repository_audit.md) | 舊版規劃的審查與遷移說明 |

開放的研究決策（Traditional DOE baseline 定義、多響應正規化方式、停止條件的 CV 偏誤、GPR 代理與最終 RSM 之間的模型不匹配）記錄於 `research_methodology.md` 與 `algorithm_decisions.md`。

## 🤝 貢獻

```bash
pip install -e ".[dev]"
pytest
```

- 🐛 回報 Bug：[開啟 Issue](https://github.com/Kelvin-Hua05/ADoRE/issues/new)
- 💡 功能建議：[Discussions](https://github.com/Kelvin-Hua05/ADoRE/discussions)

## 📝 引用

如果這個專案對你的研究有幫助，請引用：

```bibtex
@software{adore,
  author  = {Kelvin-Hua05},
  title   = {ADoRE: Adaptive Design of Experiments with Residual Enhancement},
  year    = {2026},
  url     = {https://github.com/Kelvin-Hua05/ADoRE}
}
```

## 🙏 致謝

規劃中預計建立於以下開源專案之上（目前尚未成為實際依賴）：
[`statsmodels`](https://www.statsmodels.org) · [`scikit-learn`](https://scikit-learn.org) · [`pyDOE3`](https://github.com/relf/pyDOE3) · [`SALib`](https://github.com/SALib/SALib) · [`pymoo`](https://pymoo.org) · [`DEAP`](https://github.com/DEAP/deap)

## 📄 授權

本專案以 [MIT License](LICENSE) 授權。
