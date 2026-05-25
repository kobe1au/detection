# Detection 三大创新点和实验入口

当前主线是面向时间漂移的多模态安卓恶意软件检测。默认协议来自 `config/base.yaml`：

```text
Historical pretraining: 2018-2021
Validation: 2022
Recent-year adaptation: 2023
Final test: 2024
```

论文叙事要写成一个统一主题下的三类模块化设计，而不是“同一个 drift score 直接贯穿三块”。代码里 I1 的 DBTA drift score 和 I2/I3 的 `q_drift` 不是同一个公式；它们都是 temporal drift / reliability 的实例化信号。

```text
Representative Drift-aware Budgeted Temporal Adaptation
  -> Temporal Reliability-weighted Hierarchical API-Graph Alignment
  -> Temporal Reliability-aware Multi-branch Fusion
```

## I1: Representative Drift-aware Budgeted Temporal Adaptation

I1 验证漂移信号在 budgeted adaptation / replay 中的作用。I1 实验刻意固定为 `fusion_mode: concat`、`alignment.enabled: false`、`gate.mode: fixed`，所以它不验证 alignment 或 fusion，只隔离 adaptation selection 与 replay。

核心机制：

- `train.adaptation_selection: dbta` 启用 drift-aware recent sample selection。
- `train.replay_budget_mode: selected_adapt_relative` 表示 replay 数量按已选 recent 样本数计算。
- `train.replay_budget_ratio: 0.50` 表示 `replay_count = round(selected_adapt_count * 0.50)`。
- `train.replay_strategy: drift_matched` 使用 year-class balanced anchor replay + drift-nearest historical replay。

DBTA v2 的选择流程是两阶段，不是 drift / representativeness / diversity 等权联合优化：

```text
Stage 1: drift-first candidate generation
  drift_score =
    alpha * uncertainty
  + beta  * branch_disagreement
  + gamma * prototype_distance

Stage 2: representative/diverse final selection within drift candidates
  final_selection_score mainly uses representativeness + diversity
  drift_score is retained as candidate filter and weak tie-break
```

`prototype_distance` 默认使用 nearest historical class prototype distance，同时 dump 里保留 `predicted_prototype_distance` 和 `nearest_prototype_distance`。不要再写成“只到 predicted class prototype 的距离”。

I1 baseline 语义：

| 名称 | 是否使用 2023 真标签做选择 | 含义 |
|---|---:|---|
| `random_pure` | 否 | 无标签纯随机预算基线 |
| `random_class_balanced` | 是 | oracle class-balanced random 强基线 |
| `dbta` | 否 | 漂移感知预算选样 |

I1_03 当前名为 `I1_03_dbta_no_refinement_020_dynamic_replay`。它的真实含义是 DBTA v2 without representative/diverse refinement：`dbta_candidate_top_p=1.0`、`dbta_selection_mode=topk`、`dbta_representative_weight=0`、`dbta_diversity_weight=0`。论文里建议称为 `DBTA-no-refinement`，不要写成完整 DBTA v2。

DBTA dump 语义：

- `dbta_selection.csv`：最终 selected recent samples。
- `dbta_recent_pool_scores.csv`：整个 recent pool 的 scoring dump，不是 top-p candidate pool。
- 论文和分析脚本统一使用 `drift_score`、`representativeness_score`、`diversity_gain`、`final_selection_score`。当前第一版代码不再输出旧 `selection_score` 字段。

## I2: Temporal Reliability-weighted Hierarchical API-Graph Alignment

I2 验证 alignment hierarchy。I2 组固定 I1 为 DBTA v2 20% + selected-adapt-relative drift-matched replay，并固定 gate，避免把 gate 增益混进 alignment 消融。

代码中的 hierarchy 包括：

- method-aware / adaptive cross-attention bias；
- sample-level API-Graph semantic alignment；
- class-aware semantic alignment；
- local node-token alignment；
- temporal reliability weighting via `q_time` / `q_drift`。

I2 的准确论文表述是：

```text
Temporal reliability-weighted hierarchical API-Graph alignment
```

不要写成“完整的时间条件对比学习”。当前时间信号主要通过 sample quality / time weight 调节 alignment contribution，而不是把时间作为独立监督标签重写 contrastive objective。

## I3: Temporal Reliability-aware Multi-branch Fusion

I3 验证 learned gate 如何融合 API、Graph、Joint 三个分支。I3 组固定 I1 + I2，只拆 gate evidence。

当前 full gate 输入包括：

```text
quality signals:
  q_api, q_graph, q_align, pert_api, pert_graph

temporal reliability:
  q_time, q_drift

basic time features:
  time_pos, time_recency, time_is_future, time_delta_from_history

uncertainty / availability:
  uncertainty_score, branch_disagreement, entropy, api_alive, graph_alive

confidence:
  api_confidence, graph_confidence, joint_confidence
```

注意：`time_pos` 和 `time_recency` 当前都来自 global time position，不能写成两个完全独立的新时间量。可以统称为 basic time features。

I3 消融由 `config/experiments/_manifest.yaml` 的 `i3_gate_signals` 记录每个实验实际打开的 gate signal。关键语义：

- `quality_inputs=false` 不会中和 `q_time/q_drift`；
- `quality_only` 不包含 `q_time/q_drift`；
- `temporal_reliability_only` 只打开 `q_time/q_drift`；
- `time_features_only` 只打开 basic time features；
- `full_gate` 打开 quality、temporal reliability、time features、uncertainty、confidence。

## 主实验矩阵

主表建议：

| ID | YAML | 目的 |
|---|---|---|
| M0 | `config/experiments/main_chain/M0_concat_erm.yaml` | historical concat ERM |
| M1 | `config/experiments/main_chain/M1_i1_dbta_v2_concat020.yaml` | concat + DBTA v2 20% + drift-matched replay |
| M2 | `config/experiments/main_chain/M2_i1_i2_alignment_fixed_gate020.yaml` | M1 + hierarchical alignment，gate fixed |
| M3 | `config/experiments/main_chain/M3_full_dbta_v2_020.yaml` | M2 + temporal reliability-aware learned gate |
| M4 | `config/experiments/main_chain/M4_full_random_class_balanced100_static.yaml` | full model + oracle class-balanced random 100% + static replay stress test |

Ratio sweep：

```text
2023 adaptation budget: 0% / 5% / 10% / 20% / 50% / 100%
Final test: 2024 only
```

入口：

```text
python run.py baselines
python run.py i1
python run.py replay
python run.py i2
python run.py i3
python run.py ratio
python run.py main
python run.py final
```

`python run.py final` 只跑 `config/experiments/main_chain/M3_full_dbta_v2_020.yaml`，不会展开完整 ratio sweep。

## 写作边界

- 不要写“DBTA 显著优于 random”，除非新实验支持。
- 不要把 `random_class_balanced` 写成普通 random；它是 oracle class-balanced random。
- 不要把 I1 写成已验证完整 unified framework；I1 只验证 budgeted adaptation / replay。
- 不要把 `dbta_recent_pool_scores.csv` 写成只包含 top-p candidates；它是 recent-pool scoring dump。
- 不要把旧 `selection_score` 写成论文概念；当前 DBTA dump 只保留 `final_selection_score`。
- 旧 pseudo-oracle gate / KL oracle supervision 不在当前主线中，不能写成已启用机制。
