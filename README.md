# FairForge

**FairForge** 是一个可复现的公平性审计与缓解工具包：统一口径的公平性指标、七种偏差缓解算法（pre / in / post 三族）、带反事实真值的合成有偏数据生成器，以及旗舰 **FairPareto Auto-Selector**（全"算法×超参"扫描 → 精度-公平 Pareto 前沿 → 非劣守护推荐）。

作者：**晨星**

## 特性

- **唯一硬依赖 scikit-learn**：fairlearn / optuna 为可选加速后端，缺失时自动降级到纯 numpy/sklearn 原生实现（唯一探测点 `core/backend.py`，fallback 记录进所有报告）
- **统一指标口径**：全部公平性指标为差值绝对值、越小越公平、violation ∈ [0,1]、privileged=1 / unprivileged=0；完美公平输入必须得 0（有方向性单测背书）
- **7 种缓解算法**：Reweighing、Disparate Impact Remover（pre）；Exponentiated Gradient、Grid Search（in）；Threshold Optimizer、Group Threshold、Reject Option（post）——每种都有原生兜底实现
- **FairPareto 旗舰**：Optuna（n_trials≤30、timeout 熔断）→ Pareto 前沿 → hypervolume/覆盖率打分 → 非劣守护（推荐点 violation ≤ baseline+0.01，否则回退安全侧次优点并记录事件）
- **合成有偏数据**：三种歧视机制可配置（直接歧视 δ / 代理变量 / 历史采样偏差），先抽无偏反事实 `y_cf` 再叠加机制，3 个基准场景 seed=42 可复现
- **一键落盘 benchmark.json**：accuracy、全部公平指标、backend|fallback 标记、trial 数、耗时、seed、依赖版本、环境指纹

## 快速开始

```bash
pip install -r requirements.txt          # 完整安装（含可选后端）
# 或最小离线安装：pip install -r requirements-min.txt

# 一键演示：benchmark 矩阵 + 旗舰摘要 -> benchmark.json
python examples/run_demo.py

# 仅旗舰 FairPareto -> flagship.json
python examples/flagship_demo.py

# CLI
fairforge backend                         # 查看探测到的可选后端
fairforge data --scenario mixed           # 生成并检查基准场景
fairforge run --scenario mixed --out out/benchmark.json
```

## Python API

```python
from fairforge.data.generator import generate_scenario
from fairforge.metrics.fairness import fairness_report
from fairforge.mitigation import create
from fairforge.pipeline.benchmark import run_single

ds = generate_scenario("mixed", n_per_group=600, seed=42)  # Dataset(X, A, y, y_cf)

m, y_pred, elapsed = run_single(ds, "exponentiated_gradient", seed=42)
report = fairness_report(ds.y, y_pred, ds.A)
print(report["demographic_parity_diff"])  # 越小越公平

# 旗舰
from fairforge.pipeline.flagship import run_flagship

summary = run_flagship(seed=42)  # hypervolume 提升 / 覆盖率 / 回退事件
```

## 指标口径

| 指标 | 定义 | 方向 |
|---|---|---|
| demographic_parity_diff | \|P(ŷ=1\|A=0) − P(ŷ=1\|A=1)\| | 越小越公平 |
| equal_opportunity_diff | \|TPR(A=0) − TPR(A=1)\| | 同上 |
| equalized_odds_diff | max(\|dTPR\|, \|dFPR\|) | 同上 |
| average_odds_diff | (\|dTPR\| + \|dFPR\|)/2 | 同上 |
| disparate_impact | 1 − DI（DI 对称化到 [0,1]） | 同上 |
| theil_index | 组间 Theil T | 同上 |
| calibration_by_group | 组间校准差（需 y_prob，可 NaN） | 同上 |

约定：privileged = `A==1`，unprivileged = `A==0`；`y==1` 为有利结果；除零返回 NaN，聚合跳过 NaN。

## 算法与后端

| 算法 | fairlearn 后端 | 原生兜底 |
|---|---|---|
| Reweighing | -（原生即主力） | w = P(A)·P(Y)/P(A,Y)，权重和恒等于 n |
| DisparateImpactRemover | -（原生即主力） | 组内均值对齐（repair_level 混合） |
| ExponentiatedGradient | reductions.EG + DemographicParity | 简化 reductions：加权 LR + λ←λ·exp(η·violation) + 组内阈值校正（educational-grade，与后端对拍差 <0.05） |
| GridSearch | reductions.GridSearch | 固定 λ 网格，目标 accuracy − λ·violation |
| ThresholdOptimizer | postprocessing.TO（显式 random_state） | 组内 ROC 阈值网格，约束内最大精度 |
| GroupThreshold | -（原生） | 组独立阈值匹配全局选择率 |
| RejectOption | -（原生） | 置信带 [θ,0.5] 偏向非特权组改判 |

## 基准场景

| 场景 | 机制 |
|---|---|
| mild_direct | 轻度直接歧视 + 中度代理歧视 |
| strong_sampling | 强历史采样偏差（非特权有利样本以 ρ 丢弃） |
| mixed | 三种机制混合 |

每组 ≥500 样本，seed=42 完全可复现；`y_cf` 为无偏反事实真值。

## 测试与质量

```bash
pytest -q -W ignore::UserWarning    # 75 个测试
ruff check fairforge tests examples
ruff format --check fairforge tests examples
```

包含降级矩阵测试：monkeypatch 掉 fairlearn/optuna 后整条流水线仍可运行且正确标记 fallback。

## 环境变量

`FAIRFORGE_SEED`、`FAIRFORGE_N_TRIALS`、`FAIRFORGE_LAMBDA_FAIRNESS`、`FAIRFORGE_TIMEOUT_S`、`FAIRFORGE_N_PER_GROUP`（见 `core/config.py`）。

## 旗舰 FairPareto 实测（seed=42，3 场景 × 2 约束，n=1000/组，30 trials/算法）

| 指标 | 实测 | 目标 | 达标 |
|---|---|---|---|
| hypervolume 相对最强单算法提升 | 均值约 +8%（单组 +0.5%~+15%，分场景方差大） | ≥ +10% | ❌ 未达标 |
| 前沿覆盖率 | 74%~79%（均值约 76%） | ≥ 80% | ❌ 未达标 |
| 非劣守护回退事件 | 0（推荐点始终满足 violation ≤ baseline+0.01） | 全部记录 | ✅ |

差距归因：threshold_optimizer 与 exponentiated_gradient 在约束感知的宽搜索空间下单族前沿已接近 union 前沿，组合的边际优势有限；覆盖率方面生成数据上模型精度上限约 0.85，归一化网格顶行（acc≥0.95 的 10% 格子）不可达，实际天花板约 80%。以上为诚实实测值，复现命令：`python examples/flagship_demo.py`。

## License

MIT © 晨星
