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
Temporal Continual Adaptation
  -> Class-aware API-Graph Discriminative Alignment
  -> Branch-reliability Guided Quality-aware Fusion
```

旧 temporal prototype 不再作为主创新点一。它只保留为 temporal drift diagnostics / auxiliary temporal regularization，用于分析漂移风险和可靠性。

## I1: Temporal Continual Adaptation

I1 模拟真实安全系统的周期更新：先用 2018-2021 做 historical pretraining，再用 2023 recent-year samples 做 adaptation，并混入 historical replay 缓解遗忘。

代码入口：

- `fusion/train.py` 支持 `data.adapt_csv` / `data.adapt_pt_dir`。
- `train.historical_epochs` 控制 historical phase。
- `train.adaptation_epochs` 控制 adaptation phase。
- 启用 `adapt_csv` 时，总轮数由 `historical_epochs + adaptation_epochs` 决定。
- adaptation 前加载 `best_historical_<exp>.pt`，最终保存 `best_<exp>.pt`。
- replay 采样按 `year x class` 均衡。

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
    drift_guided: true
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

I3 用 API、Graph、Joint 三个分支在训练样本上的即时 CE 构造 pseudo-oracle：

```text
oracle = softmax(-[CE_api, CE_graph, CE_joint] / temperature)
L_gate = KL(oracle || gate_weights)
```

配置入口：

```yaml
loss:
  gate_oracle_weight: 0.05
  gate_oracle_temperature: 0.5
  gate_oracle_smoothing: 0.10
  gate_oracle_start_phase: adaptation
  gate_oracle_adaptation_only: true
```

`gate_oracle_start_phase: adaptation` 避免硬编码 `historical_epochs + 1`，也避免 historical phase 早期分支未稳定时误导 gate。

## 主实验矩阵

主表建议：

| ID | 设定 | 目的 |
|---|---|---|
| B0 | zero-adapt concat baseline | 只用 2018-2021 训练，2024 测试 |
| B1 | I1 recent-year adaptation + replay | 验证 continual adaptation |
| B2 | B1 + class-aware alignment | 验证类别感知跨模态判别对齐 |
| B3 | B2 + pseudo-oracle gate | 验证质量感知多分支融合 |

ratio sweep：

```text
2023 adaptation: 5% / 10% / 20% / 100%
Final test: 2024 only
```

配置入口：

```text
config/train_2026/continual/00_zero_adapt_concat.yaml
config/train_2026/continual/01_i1_adapt_{005,010,020,100}.yaml
config/train_2026/continual/02_i1_i2_adapt_{005,010,020,100}.yaml
config/train_2026/continual/03_i1_i2_i3_adapt_{005,010,020,100}.yaml

config/train_2026/final/continual_ours_2026_adapt_{005,010,020,100}.yaml
```

这些配置使用 `resource/dataset_split_2018_2024/adapt_2023.csv` 作为 2023 recent-year adaptation pool，避免把 adaptation 数据命名成 test split。当前 replay 实现是 epoch-level mixture：每个 adaptation epoch 由 2023 recent samples 和 historical replay samples 混合训练。论文中不要写成每个 batch 固定包含 recent/replay，除非后续实现 dedicated replay batch sampler。
