# Detection 三大创新点和实验入口

当前主线改为面向持续时间漂移的多模态安卓恶意软件检测：

```text
Historical Pretraining: 2018-2021
Validation: 2022
Recent-year Adaptation: 2023 with 5% / 10% / 20% / 100% labels
Final Test: 2024
```

三点递进关系：

```text
Temporal Continual Adaptation
  -> Class-aware API-Graph Discriminative Alignment
  -> Branch-reliability Guided Quality-aware Fusion
```

旧的 temporal prototype 不再作为主创新点一。它保留为 temporal drift diagnostics / auxiliary temporal regularization，用于分析漂移风险和可靠性，而不是论文主线。

## 创新点 1：面向时间漂移的连续适应学习机制

核心问题：只用 2018-2021 训练后直接测试 2024 是干净的离线泛化设定，但真实安全系统通常会周期性吸收最近样本并更新模型。因此 I1 模拟更现实的部署协议：历史预训练后，使用 2023 的有限标注样本做 recent-year adaptation，并混入历史 replay 缓解遗忘。

代码入口：

- `fusion/train.py` 支持 `data.adapt_csv` / `data.adapt_pt_dir`。
- `train.historical_epochs` 控制 2018-2021 historical phase。
- `train.adaptation_epochs` 控制 2023 adaptation phase；总轮数由 `historical_epochs + adaptation_epochs` 决定。
- adaptation 前保存并加载 `best_historical_<exp>.pt`，最终保存 `best_<exp>.pt`。
- replay 采样按 `year x class` 均衡，避免只回放某一年或某一类。

主指标：2024 F1、Malware Recall、FNR、AUROC、AUPRC、Worst-year F1。AURC 只作为可靠性辅助指标。

## 创新点 2：类别感知的 API-Graph 跨模态判别对齐机制

核心问题：API sequence 和 call graph 不应只做同一样本语义对齐，还应在良性/恶意判别空间中同类靠近、异类分离。

代码入口：

- `fusion/mm_dataset.py` 构建 method-aware API-Graph alignment mask 和 `q_align`。
- `fusion/model.py` 使用 mask 生成 cross-attention bias 和 alignment context。
- `fusion/losses.py` 的 class-aware alignment 支持：
  - `api_i <-> graph_i`: 强正样本。
  - `api_i <-> graph_j, y_i == y_j`: 弱正样本。
  - `api_i <-> graph_j, y_i != y_j`: 负样本。

配置入口：

```yaml
loss:
  semantic_alignment_weight: 0.03
  class_aware_alignment_same_class_weight: 0.25
  class_aware_alignment_temperature: 0.2
```

建议消融：`same_class_weight = 0.0 / 0.10 / 0.25 / 0.50`。重点看 2024 F1、AUPRC、Malware Recall、高 alignment coverage 样本 F1。

## 创新点 3：伪 oracle 监督的质量感知多分支融合机制

核心问题：API、Graph、Joint 三个分支对不同 APK 的可靠性不同。仅靠最终分类 loss 间接学习 gate，容易形成固定偏好。I3 用三个分支在训练样本上的即时 CE 构造 branch-reliability pseudo-oracle，显式监督 gate 学习样本级分支可靠性。

代码入口：

- `fusion/model.py` 保留可反传的 `gate_weights_train`。
- `fusion/losses.py` 计算 `CE_api / CE_graph / CE_joint`，用 `softmax(-CE / T)` 构造 oracle。
- `gate_oracle_smoothing` 防止 oracle 过尖导致 gate 塌缩。
- `gate_oracle_start_epoch` 和 `gate_oracle_adaptation_only` 控制启用时机，避免早期分支未稳定时误导 gate。

配置入口：

```yaml
loss:
  gate_oracle_weight: 0.05
  gate_oracle_temperature: 0.5
  gate_oracle_smoothing: 0.10
  gate_oracle_start_epoch: 61
  gate_oracle_adaptation_only: true
```

重点看 2024 F1、Malware Recall、低 API/Graph 质量样本 F1、gate weight by correctness/year/quality。

## 主实验矩阵

推荐主表：

| ID | 设定 | 目的 |
|---|---|---|
| B0 | zero-adapt concat baseline | 只用 2018-2021 训练，2024 测试 |
| B1 | I1 recent-year adaptation + replay | 验证 continual adaptation |
| B2 | B1 + class-aware alignment | 验证类别感知跨模态判别对齐 |
| B3 | B2 + pseudo-oracle gate | 验证质量感知多分支融合 |

adaptation ratio 固定报告：

```text
2023 adaptation: 5% / 10% / 20% / 100%
Final test: 2024 only
```

当前入口：

```text
config/train_2026/final/continual_ours_2026_adapt_005.yaml
config/train_2026/final/continual_ours_2026_adapt_010.yaml
config/train_2026/final/continual_ours_2026_adapt_020.yaml
config/train_2026/final/continual_ours_2026_adapt_100.yaml
```

旧 `config/train_2026/i1_temporal/*` 只用于 temporal prototype 诊断和历史对照，不再作为新论文主线。
