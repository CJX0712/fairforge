# FairForge 架构文档

作者：晨星
版本：0.1.0

## 1. 设计原则

1. **单向无环依赖**：`cli / pipeline / examples → hpo → mitigation → metrics → core`，`data → core`。任何下层模块不得反向导入上层。
2. **唯一探测点**：可选依赖（fairlearn / optuna）的可用性只在 `core/backend.py` 判定一次（缓存）。其余模块通过 `get_backend(name, requested)` 解析；fallback 时记录一行日志并在结果对象上标记 `used_fallback`。
3. **统一指标口径**：诊断、约束目标、HPO 目标、benchmark 列全部使用 `metrics/fairness.py` 的同一组实现，杜绝口径漂移。
4. **干净环境可复现**：唯一硬依赖 scikit-learn；`requirements-min.txt` 离线最小安装即可跑通全部功能（原生兜底）。

## 2. 模块结构

```
fairforge/
  core/      types.py(Dataset/RunResult)  errors.py(E100~E500)  config.py(FAIRFORGE_*)  backend.py(唯一探测点)
  data/      generator.py(三机制合成有偏数据+y_cf)  adult.py(可选,失败返回 None)
  metrics/   fairness.py(7 指标统一口径)  performance.py
  mitigation/ base.py(Mitigator 基类+注册表)  pre/  inproc/  post/
  hpo/       tuner.py(Optuna 封装+随机搜索兜底)  selector.py(FairPareto 旗舰)
  pipeline/  benchmark.py(矩阵跑批+落盘)  flagship.py(场景×约束编排)
  cli.py     fairforge run / data / backend
```

## 3. 数据层

`generate_biased_data` 流程（seed 固定 42 可复现）：

1. 抽取 `A ~ Bernoulli(0.5)`、潜变量 skill；
2. **先抽无偏反事实标签** `y_cf ~ Bernoulli(sigmoid(1.2·skill+ε))`；
3. 观测 `y` = `y_cf` 的 logit 上叠加：直接歧视 `δ·(2A−1)` + 代理路径 `0.9·x_proxy`（`x_proxy = proxy_strength·(2A−1)+ε`，作为特征 X 的第 0 列可见）；
4. 历史采样偏差：非特权组中 `y_cf==1` 的样本以概率 ρ 丢弃（超采样保证每组仍 ≥ n_per_group）。

三个内置场景：`mild_direct`（δ=0.9, proxy=1.3）、`strong_sampling`（ρ=0.7）、`mixed`（δ=0.7, proxy=1.0, ρ=0.4）。

## 4. 指标层

统一约定：privileged=`A==1`、unprivileged=`A==0`、有利结果=`y==1`；所有 violation 为差值绝对值 ∈ [0,1]，越小越公平，完美公平=0；除零 → NaN，聚合跳过 NaN；`disparate_impact` 将 DI 对称化（min(ratio, 1/ratio)）后报 `1−DI` 以保证落在 [0,1]；`theil_index` 为**组间** Theil T（组选择率相等时恰为 0）。方向性由单测锁定。

## 5. 缓解层

- `Mitigator` 基类：`fit(Dataset)` + `predict(X, A)`（pre 类为 `transform`），构造时解析 backend 并记录 `resolved_backend / used_fallback`。
- 注册表：算法模块被 import 时经 `@register` 装饰器自动注册；`mitigation/__init__.py` 负责全部导入，保证 `import fairforge` 后注册表完整。
- 七个算法见 README 表格。原生 ExponentiatedGradient 为 educational-grade 简化 reductions：样本权重按符号差指数更新（λ←λ·exp(η·gap)）+ 组内分位数阈值校正（等率解与 ±eps 边界解中按精度择优，全部满足 violation≤eps）；与 fairlearn 后端做对拍软断言（工作点校准后 |Δviolation|<0.05）。

## 6. HPO 层与旗舰 FairPareto

`Tuner`：Optuna TPESampler(seed) 最大化 `accuracy − λ·violation`，n_trials≤30、timeout 熔断；optuna 缺失时降级为种子化随机搜索（同接口同输出形状）。每个 trial 的 (params, value) 全量记录供 Pareto 分析。

`FairParetoSelector.fit_select(dataset)` 五步：

1. baseline = 朴素逻辑回归的 (accuracy, violation)；
2. 每个算法族跑 HPO，收集全部 (acc, viol) 点；
3. 构建精度-公平 Pareto 前沿（支配关系去重）；
4. **非劣守护**：目标最优点须满足 `violation ≤ baseline_violation + 0.01`，否则回退前沿上 violation 最小的安全点并记 `fallback_event`；
5. 打分：hypervolume（参考点 (acc→1−acc, viol→1) 的精确 2D 计算）对比最强单算法自身前沿的 hypervolume；覆盖率 = 归一化目标空间 10×10 网格中被前沿支配的格点占比。

验收门槛（3 场景 × 2 约束共 6 组）：hypervolume 相对最强单算法 ≥ +10%、覆盖率 ≥ 80%、回退事件全部记录。实测值无论达标与否均如实写入 benchmark.json。

## 7. 降级矩阵

| 场景 | 行为 |
|---|---|
| fairlearn 缺失 | 所有算法走原生实现，`used_fallback=True` |
| optuna 缺失 | Tuner 走种子化随机搜索 |
| 网络不可达（Adult） | `load_adult()` 返回 None，日志一行跳过 |

三者均有自动化测试（monkeypatch 探测函数模拟缺失环境）。

## 8. 复现指南

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
pytest -q -W ignore::UserWarning        # 75 passed
python examples/run_demo.py             # 落盘 benchmark.json
ruff check fairforge tests examples && ruff format --check fairforge tests examples
```

已知约束：Windows 下 `n_jobs` 锁死为 1（multiprocessing 限制）；sklearn ≥1.7 不接受 `multi_class` 参数，代码只使用稳定参数面。
