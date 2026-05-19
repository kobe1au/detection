# 2026 实验计划：base + override

所有实验使用同一个 base：

```text
config/base.yaml
```

base 只保留数据路径、模型默认值和训练超参。实验差异全部放在 `config/train_2026/**.yaml` override 中。

## 数据划分

主实验使用：

```text
Train: 2018-2021
Val:   2022
Test:  2023
Test:  2024
Test:  2023-2024 combined
```

训练时仍使用 `test.csv` 汇总测试集，训练日志会额外输出 per-year AUT。

## I1：跨年份-标签-子簇原型演化时间底座

| ID | YAML | 设置 | 目的 |
|---|---|---|---|
| I1-0 | `config/train_2026/i1_temporal/00_erm_concat.yaml` | 无时间约束 | 时间漂移基线 |
| I1-1 | `config/train_2026/i1_temporal/01_proto_current.yaml` | 当前 year-label-cluster 原型 | 验证跨 batch 多子簇原型记忆 |
| I1-2 | `config/train_2026/i1_temporal/02_proto_current_010.yaml` | 当前原型权重 0.10 | 验证权重敏感性 |
| I1-3 | `config/train_2026/i1_temporal/03_proto_future_weak.yaml` | 当前 + 未来原型 | 验证原型演化约束 |
| I1-4 | `config/train_2026/i1_temporal/04_ours_fixed_scaffold_erm.yaml` | ours fixed，无时间约束 | 控制最终骨架变量 |
| I1-5 | `config/train_2026/i1_temporal/05_ours_fixed_proto_trajectory.yaml` | ours fixed + 当前/未来原型 | 验证时间底座可迁移到最终骨架 |

重点看 `train_proto_current`、`train_proto_future`、`latest_F1`、`worst_F1`、`AUT_F1`。

## I2：时间引导的 API-Graph 语义对齐

I2 固定使用 I1 的时间底座，且 gate 使用 fixed average，避免门控影响对齐模块归因。

| ID | YAML | 设置 | 目的 |
|---|---|---|---|
| I2-0 | `config/train_2026/i2_alignment/00_temporal_concat.yaml` | concat | 无结构交互基线 |
| I2-1 | `config/train_2026/i2_alignment/01_cross_attention.yaml` | 普通 cross-attention | 验证跨模态注意力 |
| I2-2 | `config/train_2026/i2_alignment/02_ours_fixed_no_alignment.yaml` | ours fixed，无 alignment/semantic loss | 控制 ours 骨架变量 |
| I2-3 | `config/train_2026/i2_alignment/03_semantic_alignment.yaml` | semantic loss only | 验证全局语义对齐损失 |
| I2-4 | `config/train_2026/i2_alignment/04_method_aware_context.yaml` | method-aware bias/context only | 验证结构先验 |
| I2-5 | `config/train_2026/i2_alignment/05_temporal_guided_alignment.yaml` | drift-guided bias/context + semantic loss | 验证完整 I2 |

重点看 `train_alignment`、`alignment_coverage`、`alignment_density`、`alignment_temporal_drift` 和按年份 F1。

## I3：漂移感知可靠性门控

I3 固定使用完整 I1 + I2。

| ID | YAML | gate.mode | 质量信号 | 漂移信号 | 目的 |
|---|---|---|---|---|---|
| I3-0 | `config/train_2026/i3_fusion/00_fixed_no_gate.yaml` | fixed | 0 | 0 | 真正无 learned gate |
| I3-1 | `config/train_2026/i3_fusion/01_learned_gate_no_reliability.yaml` | learned | 0 | 0 | 只验证 learned gate 结构 |
| I3-2 | `config/train_2026/i3_fusion/02_quality_gate.yaml` | learned | 1 | 0 | 只验证质量感知 |
| I3-3 | `config/train_2026/i3_fusion/03_drift_gate.yaml` | learned | 0 | 1 | 只验证漂移感知 |
| I3-4 | `config/train_2026/i3_fusion/04_quality_drift_gate.yaml` | learned | 1 | 1 | 完整可靠性门控 |

重点看 `train_gate_api`、`train_gate_graph`、`train_gate_joint` 是否随年份、漂移和模态质量变化。

## Baselines

| ID | YAML | 类型 |
|---|---|---|
| B0 | `config/train_2026/baselines/00_api_only.yaml` | API 单模态 |
| B1 | `config/train_2026/baselines/01_graph_only.yaml` | Graph 单模态 |
| B2 | `config/train_2026/baselines/02_concat_erm.yaml` | concat ERM |
| B3 | `config/train_2026/baselines/03_cross_attention.yaml` | 普通 cross-attention |

## Final Ablation

| ID | YAML | 目的 |
|---|---|---|
| F0 | `config/train_2026/final/ours_2026_no_future.yaml` | 去掉未来原型 |
| F1 | `config/train_2026/final/ours_2026_no_semantic_align.yaml` | 去掉语义对齐模块 |
| F2 | `config/train_2026/final/ours_2026_no_reliability_gate.yaml` | fixed average，去掉 learned reliability gate |
| F3 | `config/train_2026/final/ours_2026.yaml` | 完整模型 |

## 运行

```bash
bash scripts/run_train_2026.sh all
```

单独运行：

```bash
bash scripts/run_train_2026.sh baselines
bash scripts/run_train_2026.sh i1
bash scripts/run_train_2026.sh i2
bash scripts/run_train_2026.sh i3
bash scripts/run_train_2026.sh final
```
