# Detection 三大创新点和实验入口

当前方法按递进关系组织：

```text
跨年份-标签-子簇原型演化时间底座
  -> 时间引导的 API-Graph 语义对齐
  -> 漂移感知的多模态可靠性门控
```

## 创新点 1：跨年份-标签-子簇原型演化的时间漂移建模

核心逻辑：

- `fusion/prototypes.py` 的 `TemporalPrototypeMemory` 跨 batch 维护 `year-label-cluster` 原型，每个年份-标签单元保留多个子原型以刻画 APK 类内多峰分布。
- 原型损失现在约束最终融合表征经 `temporal_feature_proj` 投影后的 128 维时间表征，而不是只约束 API/Graph 平均向量。
- 通过 `P(t,c,k) - P(t-1,c,k)` 预测未来年份类别子簇原型，并用 future prototype consistency 约束当前表征。
- 同一 prototype memory 还能在推理阶段用预测类别概率估计 `temporal_drift_score`，不依赖测试标签。

实验入口：

| ID | YAML | 目的 |
|---|---|---|
| I1-0 | `config/train_2026/i1_temporal/00_erm_concat.yaml` | 无时间模块的 concat ERM |
| I1-1 | `config/train_2026/i1_temporal/01_proto_current.yaml` | 当前 year-label-cluster 原型 |
| I1-2 | `config/train_2026/i1_temporal/02_proto_current_010.yaml` | 当前原型权重敏感性 |
| I1-3 | `config/train_2026/i1_temporal/03_proto_future_weak.yaml` | 当前原型 + 未来原型 |
| I1-4 | `config/train_2026/i1_temporal/04_ours_fixed_scaffold_erm.yaml` | 最终 ours 骨架中无时间底座 |
| I1-5 | `config/train_2026/i1_temporal/05_ours_fixed_proto_trajectory.yaml` | 最终 ours 骨架中验证时间底座 |

重点看 `train_proto_current`、`train_proto_future`、`latest_F1`、`worst_F1`、`AUT_F1`。

## 创新点 2：时间引导的 API-Graph 语义对齐

核心逻辑：

- `fusion/mm_dataset.py` 构建 method-API alignment mask 和 `q_align`。
- `fusion/model.py` 用 method-aware mask 产生 cross-attention bias 和 alignment context refinement。
- `fusion/losses.py` 用 `semantic_alignment_weight` 约束 API/Graph paired representation。
- 新增 drift guidance：创新点 1 的 prototype drift 会调低高漂移样本的对齐强度，避免错误模态互相污染。

实验入口：

| ID | YAML | 目的 |
|---|---|---|
| I2-0 | `config/train_2026/i2_alignment/00_temporal_concat.yaml` | 时间底座 + concat |
| I2-1 | `config/train_2026/i2_alignment/01_cross_attention.yaml` | 时间底座 + 普通 cross-attention |
| I2-2 | `config/train_2026/i2_alignment/02_ours_fixed_no_alignment.yaml` | 相同 ours fixed 骨架，无对齐 |
| I2-3 | `config/train_2026/i2_alignment/03_semantic_alignment.yaml` | 只加 sample-level semantic loss |
| I2-4 | `config/train_2026/i2_alignment/04_method_aware_context.yaml` | 只加 method-aware bias/context |
| I2-5 | `config/train_2026/i2_alignment/05_temporal_guided_alignment.yaml` | temporal-guided method-aware semantic alignment |

重点看 `train_alignment`、`alignment_coverage`、`alignment_density`、`alignment_temporal_drift` 和按年份 F1。

## 创新点 3：漂移感知的多模态可靠性门控

核心逻辑：

- `fusion/model.py` 的 gate 输入包括模态质量、API/Graph/Joint 分歧、联合分支熵和 prototype temporal drift。
- `gate.mode=fixed` 是真正的固定三分支平均，用于 no learned gate 消融。
- `gate.mode=learned` 启用 `TriBranchGate`，动态输出 `[w_api, w_graph, w_joint]`。
- `quality_inputs` 和 `drift_inputs` 分别控制质量信号与漂移信号是否进入 learned gate。

实验入口：

| ID | YAML | 目的 |
|---|---|---|
| I3-0 | `config/train_2026/i3_fusion/00_fixed_no_gate.yaml` | 固定三分支平均，无 learned gate |
| I3-1 | `config/train_2026/i3_fusion/01_learned_gate_no_reliability.yaml` | learned gate，但无质量/漂移输入 |
| I3-2 | `config/train_2026/i3_fusion/02_quality_gate.yaml` | 只输入质量信号 |
| I3-3 | `config/train_2026/i3_fusion/03_drift_gate.yaml` | 只输入漂移信号 |
| I3-4 | `config/train_2026/i3_fusion/04_quality_drift_gate.yaml` | 完整质量 + 漂移门控 |

重点看 `train_gate_api`、`train_gate_graph`、`train_gate_joint`、`train_drift` 和 per-year 测试结果。

## 最终模型

完整模型：

```text
config/train_2026/final/ours_2026.yaml
```

最终消融：

| ID | YAML | 目的 |
|---|---|---|
| F0 | `config/train_2026/final/ours_2026_no_future.yaml` | 去掉未来原型约束 |
| F1 | `config/train_2026/final/ours_2026_no_semantic_align.yaml` | 去掉语义对齐模块 |
| F2 | `config/train_2026/final/ours_2026_no_reliability_gate.yaml` | 固定平均，去掉 learned reliability gate |
| F3 | `config/train_2026/final/ours_2026.yaml` | 完整模型 |

训练协议：

- 所有实验从头训练。
- 主划分：`2018-2021 train / 2022 val / 2023-2024 test`。
- 主报告：`2023`、`2024`、`2023-2024 combined`，同时报告 `AUT_F1`、`latest_F1`、`worst_F1`、`AURC`。
