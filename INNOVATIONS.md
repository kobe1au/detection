# Detection 三大创新点和实验入口

当前主线是面向持续时间漂移的多模态安卓恶意软件检测：

```text
Historical pretraining: 2018-2021
Validation: 2022
Recent-year adaptation: 2023 with 5% / 10% / 20% / 100% labels
Final test: 2024
```

三点递进关系：

```text
DBTA Budgeted Temporal Adaptation
  -> Class-aware API-Graph Discriminative Alignment
  -> Branch-reliability Guided Quality-aware Fusion
```

旧 temporal prototype 不再作为主创新点一。它只保留为 temporal drift diagnostics / auxiliary temporal regularization，用于分析漂移风险和可靠性。

## I1: DBTA: Drift-aware Budgeted Temporal Adaptation

I1 不再只是“随机取 2023 的 5%/10%/20% 样本做 fine-tuning”。当前主线是 DBTA：先用 2018-2021 historical checkpoint 对 2023 recent pool 做无标签漂移打分，再在固定标注预算下选择最有时间漂移信息的样本，最后用选中样本的标签做 adaptation。

代码入口：

- `fusion/train.py` 支持 `data.adapt_csv` / `data.adapt_pt_dir`。
- `train.historical_epochs` 控制 historical phase。
- `train.adaptation_epochs` 控制 adaptation phase。
- `train.adaptation_selection: dbta` 启用漂移感知预算选择。
- 启用 `adapt_csv` 时，总轮数由 `historical_epochs + adaptation_epochs` 决定。
- adaptation 前加载 `best_historical_<exp>.pt`，最终保存 `best_<exp>.pt`。
- `replay_strategy: drift_matched` 使用 year-class anchor replay + drift-nearest historical replay。

DBTA drift score 使用 historical model 对 2023 pool 的预测信号，选择阶段不使用 2023 真标签：

```text
drift_score(x) =
  alpha * uncertainty(x)
+ beta  * branch_disagreement(x)
+ gamma * prototype_distance(x)
```

其中 `prototype_distance` 来自 2018-2021 historical class prototypes；预算选择默认按 predicted label 做平衡，避免 top-k 全落在单一预测类别。

主指标：2024 F1、Malware Recall、FNR、AUROC、AUPRC、Worst-year F1。AURC 只作为可靠性辅助指标。

## I2: Class-aware API-Graph Discriminative Alignment

I2 不只让同一样本的 API 和 Graph 表征靠近，还利用标签构造类别级跨模态关系：

- `api_i <-> graph_i`: 强正样本。
- `api_i <-> graph_j, y_i == y_j`: 弱正样本。
- `api_i <-> graph_j, y_i != y_j`: 负样本。

配置入口：

```yaml
model:
  alignment:
    enabled: true
    adaptive_bias: true
    penalty_scale: 0.5
    bonus_scale: 1.0
    context_scale: 0.35
loss:
  semantic_alignment_weight: 0.03
  class_aware_alignment_same_class_weight: 0.25
  class_aware_alignment_temperature: 0.2
```

`penalty_scale` 用于 method-aware attention bias 的未对齐位置抑制；类别级负样本分离由 class-aware contrastive objective 提供。

## I3: Branch-reliability Guided Quality-aware Fusion

I3 当前实现是 API、Graph、Joint 三分支门控融合，gate 输入由模态质量、分支不确定性和模态可用性组成：

```text
gate_inputs =
  [q_api, q_graph, q_align, pert_api, pert_graph,
   branch_disagreement, entropy, api_alive, graph_alive]
```

配置入口：

```yaml
model:
  gate:
    mode: learned
    quality_inputs: true
    uncertainty_inputs: true
    detach: true
```

旧 pseudo-oracle gate 配置已经移出主线；当前不能把 KL oracle supervision 写成已启用机制。

## 主实验矩阵

主表建议：

| ID | 设定 | 目的 |
|---|---|---|
| B0 | zero-adapt concat baseline | 只用 2018-2021 训练，2024 测试 |
| B1 | I1 DBTA + drift-matched replay | 验证漂移感知预算适应 |
| B2 | B1 + class-aware alignment | 验证类别感知跨模态判别对齐 |
| B3 | B2 + quality/uncertainty tri-branch gate | 验证质量感知多分支融合 |

ratio sweep：

```text
2023 adaptation: 5% / 10% / 20% / 100%
Final test: 2024 only
```

配置入口：

```text
config/train_2026/main_chain/00_zero_adapt_concat.yaml
config/train_2026/main_chain/01_i1_adapt_020.yaml
config/train_2026/main_chain/02_i1_i2_adapt_020.yaml
config/train_2026/main_chain/03_i1_i2_i3_adapt_020.yaml

config/train_2026/ratio_sweep/*full_adapt*.yaml
```

当前 YAML 仍使用 `results/labels/test_2023.csv` / `/pts/test` 作为 2023 recent-year adaptation pool 的路径命名；论文和最终归档配置应改成 `adapt_2023` 语义命名，避免被误读成测试集泄漏。当前 replay 实现是 epoch-level mixture：每个 adaptation epoch 由选中的 2023 recent samples 和 historical replay samples 混合训练。论文中不要写成每个 batch 固定包含 recent/replay，除非后续实现 dedicated replay batch sampler。
