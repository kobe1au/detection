# Source-Aware APK Evidence Graph Malware Detection

本项目实现了一个面向 Android 恶意软件检测的鲁棒性检测框架，核心方法基于 **源感知 APK 异构证据图** 与 **可靠性感知多视图 KL 一致性学习**。

当前主线已经不再是 contrastive learning / InfoNCE。当前训练目标是：

```text
交叉熵分类损失
+ 可选的 clean / degraded 视图 KL 一致性约束
+ 可靠性感知的一致性权重
```

当前支持的 loss 模式为：

```text
ce_only      仅使用交叉熵分类损失
plain_kl     交叉熵 + 普通 clean/degraded KL 一致性
compact_kl   交叉熵 + 可靠性感知 clean/degraded KL 一致性
```

运行环境使用 Python 3.10 或 3.11；当前固定的 `torch==2.2.0` 不支持本机默认的 Python 3.13。

---

## 1. 方法概述

本项目围绕三个核心设计展开。

### 1.1 源感知 APK 异构证据图建模

APK 被建模为一个带有节点类型、边类型、证据源和质量信息的异构证据图。

图中融合的证据包括：

```text
代码侧证据
Manifest 声明证据
派生风险语义节点
字符串 / 静态 hint 证据
代码与 Manifest 之间的对齐证据
```

该设计用于替代只依赖单一 API 序列、单一权限特征或单一 Manifest 特征的传统检测方式。

### 1.2 可靠性感知多视图 KL 一致性学习

模型在 clean graph view 和 degraded graph view 之间施加预测一致性约束。

当前方法不使用 InfoNCE，也不依赖 batch 内负样本，而是使用 KL consistency。

其中：

```text
plain_kl   使用普通 clean/degraded KL 一致性
compact_kl 根据 API / graph / Manifest 可靠性与扰动类型调整一致性权重
```

这样可以避免 batch size 过小导致 contrastive loss 不稳定的问题，也更贴合论文中的“可靠性感知一致性学习”主线。

### 1.3 可靠性与冲突感知的 latent fusion

模型会把多类证据编码成 latent tokens，例如：

```text
method token
API-family token
permission token
component token
risk token
string-hint token
global token
```

融合时会考虑：

```text
代码侧可靠性
Manifest 侧可靠性
证据源 bias
代码-Manifest 冲突
不同证据 token 的 availability
```

目标是降低模型对 Manifest shortcut 的过度依赖，使模型在 Manifest 被污染、打乱、缺失或隐藏污染标记时仍然保持较好鲁棒性。

---

## 2. 当前主线状态

当前有效主线为：

```text
Source-aware AEG
+ reliability-aware latent fusion
+ CE / Plain-KL / Compact-KL objectives
+ robust validation
+ synthetic robust evaluation
+ Obfuscapk external evaluation
```

旧的 contrastive 主线已经不再使用。

不要再把当前方法描述为：

```text
InfoNCE
multi-view contrastive learning
clean-degraded contrast
source-level contrast
cross-source contrast
counterfactual contrastive loss
```

应统一描述为：

```text
多视图 KL 一致性学习
可靠性感知一致性权重
Manifest shortcut suppression
源感知异构证据图建模
真实混淆外部泛化评估
```

---

## 3. 仓库结构

```text
extract/
  extract_graph_api.py                  DEX / API / 调用图提取逻辑

fusion/
  aeg_builder.py                        构建 APK Evidence Graph payload
  config_utils.py                       YAML 配置加载与 base 继承
  constants.py                          节点类型、边类型、证据源、视图定义
  dataset.py                            AEG Dataset、增强、PyG batch collate
  io_utils.py                           安全加载 AEG payload 和 checkpoint
  losses.py                             CE + Plain-KL / Compact-KL 一致性损失
  manifest_features.py                  Manifest 解析、词表、向量化
  model.py                              类型图编码器 + 可靠性感知 latent fusion
  perturbations.py                      clean/degraded graph view 扰动逻辑
  train.py                              训练、验证、鲁棒评估、diagnostics 输出

scripts/
  build_aeg_pts_direct.py               APK -> AEG .pt 构建脚本
  build_obfuscapk_label_csvs.py         构建 Obfuscapk 外部评估标签 CSV
  evaluate_aeg_checkpoint.py            使用已训练 checkpoint 评估外部 PT 场景
  summarize_obfuscapk_pairs.py          clean-vs-obfuscated 配对 flip-rate 统计
  validate_aeg_pts.py                   PT/schema/contract 校验
  summarize_aeg_diagnostics.py          diagnostics 汇总

config/
  extract/
    extract_aeg.yaml
    extract_aeg_vocab_train_only.yaml
    extract_aeg_val_test.yaml
    extract_aeg_train_only.yaml
    extract_aeg_behavior_hints.yaml

  extract_obfuscapk.yaml
  eval_obfuscapk.yaml

  experiments/aeg_robust/
    base.yaml                           中性 AEG-only 公共模板
    main/                               full compact-KL 主方法 seed42/43/44
    stage1/                             创新点1：AEG 表示与结构消融
    stage2/                             创新点2：可靠性 / 冲突感知 fusion
    stage3/                             创新点3：Plain-KL / Compact-KL

tests/
  test_aeg_smoke.py                     payload / model / loss / config / runner smoke tests
```

---

## 4. 构建 AEG PT 数据

当前 AEG schema 为 v7。v7 将可靠性重新定义为提取完整性，不再使用 API 数量、图规模或 Manifest 词表覆盖率作为质量捷径；旧 schema PT 必须重新构建。

### 4.1 使用 train split 重建 Manifest 词表并构建 PT

```bash
python scripts/build_aeg_pts_direct.py \
  --config config/extract/extract_aeg.yaml \
  --rebuild-vocab \
  --no-resume \
  --workers 8
```

### 4.2 断点续跑，不重建词表

```bash
python scripts/build_aeg_pts_direct.py \
  --config config/extract/extract_aeg.yaml \
  --no-rebuild-vocab \
  --resume \
  --workers 8
```

### 4.3 只构建 train-derived Manifest 词表

如果磁盘空间或时间不够，可以先只构建词表：

```bash
python scripts/build_aeg_pts_direct.py \
  --config config/extract/extract_aeg_vocab_train_only.yaml \
  --rebuild-vocab \
  --vocab-only \
  --workers 8
```

### 4.4 使用冻结词表构建 val/test

```bash
python scripts/build_aeg_pts_direct.py \
  --config config/extract/extract_aeg_val_test.yaml \
  --no-rebuild-vocab \
  --resume \
  --workers 8
```

### 4.5 后续单独构建 train PT

```bash
python scripts/build_aeg_pts_direct.py \
  --config config/extract/extract_aeg_train_only.yaml \
  --no-rebuild-vocab \
  --resume \
  --workers 8
```

---

## 5. 校验 AEG PT

训练前建议先抽样校验 PT：

```bash
python scripts/validate_aeg_pts.py \
  --config config/extract/extract_aeg.yaml \
  --sample-per-split 100
```

完整校验：

```bash
python scripts/validate_aeg_pts.py \
  --config config/extract/extract_aeg.yaml \
  --all
```

校验内容包括：

```text
CSV/PT id 是否一致
节点特征维度是否合法
schema version 是否匹配
payload contract 是否匹配
必要 tensor 字段是否存在
train/val/test 是否存在样本泄漏
package_name 是否跨 split 重复
```

---

## 6. 实验配置

共享实验模板为：

```text
config/experiments/aeg_robust/base.yaml
```

需要先修改其中的数据路径，例如：

```yaml
data:
  train:
    pt_dir: D:/pts_aeg/train
    csv: results/labels/train.csv

  val:
    pt_dir: D:/pts_aeg/val
    csv: results/labels/val.csv

  test:
    pt_dir: D:/pts_aeg/test
    csv: results/labels/test.csv
```

`base.yaml` 是严格中性基线：

```yaml
train:
  checkpoint_metric: macro_f1

robust:
  train_aug: false

loss:
  mode: ce_only

model:
  fusion_mode: masked_mean
  reliability_bias_weight: 0.0
  conflict_bias_weight: 0.0
  source_bias_weight: 0.0
```

创新点2配置显式切换到 latent fusion；创新点3配置才开启训练扰动和 KL：

```yaml
robust:
  train_aug: true

loss:
  mode: compact_kl
  consistency_weight: 0.05

model:
  hidden_dim: 128
  layers: 2
  dropout: 0.15
  num_latents: 16
  fusion_mode: latent
  num_classes: 2

  use_relation_types: true
  use_node_types: true
  use_node_source: true
  use_edge_source: true
  use_node_quality: true
  use_edge_quality: true

  reliability_bias_weight: 1.0
  conflict_bias_weight: 0.5
  source_bias_weight: 1.0

  allow_node_dim_adapt: false
```

---

## 7. 训练主方法

### 7.1 训练 seed42 主方法

```bash
python -m fusion.train \
  --config config/experiments/aeg_robust/main/full_compact_kl_seed42.yaml
```

### 7.2 使用 runner 训练主方法

```bash
python run.py final
```

等价 alias：

```bash
python run.py ours
python run.py full
python run.py compact
```

---

## 8. 实验 runner

`run.py` 会在以下目录下解析实验配置：

```text
config/experiments/aeg_robust/
```

可用单实验入口：

```bash
python run.py final
python run.py aeg_only
python run.py fusion
python run.py plain_kl
python run.py compact_kl
```

可用实验组：

```bash
python run.py main
python run.py full_seeds
python run.py stage1
python run.py stage2
python run.py stage3
python run.py all
```

dry-run 查看将要运行的配置：

```bash
python run.py all --dry-run
```

当前实验组语义：

```text
main        full compact-KL seed42
full_seeds  full compact-KL seed42/43/44
stage1      AEG-only CE：单源基线与图结构消融，排除创新点2/3
stage2      完整 AEG + CE：latent / reliability / conflict / source fusion
stage3      完整 AEG + 完整 fusion：CE / Plain-KL / Compact-KL
all         按 stage1 -> stage2 -> stage3 -> full seeds 运行
```

旧组名 `r1_graph`、`r3_fusion`、`loss` 仍作为兼容 alias，分别映射到 stage1、stage2、stage3。

---

## 9. 核心实验设计

### 9.1 主方法实验

```bash
python run.py final
```

主配置：

```text
config/experiments/aeg_robust/main/full_compact_kl_seed42.yaml
```

三随机种子实验：

```bash
python run.py full_seeds
```

对应配置：

```text
config/experiments/aeg_robust/main/full_compact_kl_seed42.yaml
config/experiments/aeg_robust/main/full_compact_kl_seed43.yaml
config/experiments/aeg_robust/main/full_compact_kl_seed44.yaml
```

### 9.2 Stage 1：单独验证源感知 AEG

```bash
python run.py stage1
```

比较：

```text
API-only CE
Graph-only CE
Manifest-only CE
AEG-only CE
AEG no source metadata CE
AEG no relation types CE
AEG no quality CE
AEG no alignment CE
AEG no risk CE
```

所有配置必须保持：

```text
CE-only
masked_mean
无训练扰动
无 reliability / conflict / source fusion bias
```

严格解释边界：当前 Graph-only 分支仍来自敏感 API 中心子图，因此它验证的是“只保留图结构/方法证据后的增益”，不是完全独立于 API 选择过程的纯调用图基线。论文中不得把它描述为 API-independent graph baseline。No-alignment 同时移除显式跨源对齐边和 Manifest-to-risk 边，但保留代码侧 risk 表达；No-risk 则删除全部 risk 节点。

### 9.3 Stage 2：引入可靠性与冲突感知 fusion

```bash
python run.py stage2
```

包括：

```text
latent content only
reliability only
conflict only
source-score bias only
no reliability
no conflict
no source-score bias
full fusion
```

Stage 2 仍然只使用 CE 且不使用训练扰动，确保鲁棒变化来自 fusion 本身。

这里的 `conflict` 实际是跨源语义分歧代理，不是可观测的逻辑矛盾。由于两源证据可能天然互补，只有当 conflict 消融和盲扰动实验共同证明其有效时，才能将它作为创新点；否则应降级为诊断量，不能过度宣称“冲突检测”。

### 9.4 Stage 3：引入可靠性感知多视图 KL

```bash
python run.py stage3
```

比较：

```text
Full fusion CE
Full fusion Plain-KL
Full fusion Compact-KL
Compact-KL w=0.02
Compact-KL w=0.10
```

核心结论应检验：

```text
Plain-KL 是否提升退化视图稳定性；
Compact-KL 是否优于等权 Plain-KL；
consistency weight 是否存在稳定区间。
```

---

## 10. 内部鲁棒评估

默认 robust views 包括：

```text
api_degraded
graph_degraded
api_graph_degraded
manifest_noisy
manifest_shuffled
manifest_noisy_blind
manifest_shuffled_blind
api_missing
graph_missing
manifest_missing
all_degraded
```

建议报告指标：

```text
Clean Macro-F1
Robust Macro-F1 per view
Robust average Macro-F1
AUC
Average Precision
ECE
Brier Score
Robustness Drop
```

递进式主实验统一使用 clean validation Macro-F1 选择 checkpoint，避免鲁棒场景泄漏进模型选择。`robust_composite` 只适合额外的部署导向实验，不应与递进式主表直接比较。

---

## 11. 训练输出

每次训练通常输出：

```text
best.pt
experiment_metadata.json
history.csv
summary.json
diagnostics_val.csv
diagnostics_test_clean.csv
diagnostics_test_<view>_<strength>.csv
```

其中：

```text
summary.json               最终指标摘要
history.csv                每个 epoch 的训练和验证记录
experiment_metadata.json   实验配置、环境、git 信息、数据集统计、best epoch、最终结果
diagnostics_*.csv          样本级预测、可靠性、attention、扰动信息
```

diagnostics 中建议重点分析：

```text
sid
label
pred
prob_malware
q_api
q_graph
q_manifest
code_reliability
manifest_reliability
code_manifest_conflict
attn_method
attn_api_family
attn_permission
attn_component
attn_risk
attn_string_hint
attn_global
```

---

## 12. diagnostics 汇总

```bash
python scripts/summarize_aeg_diagnostics.py \
  --input-dir results/aeg_robust/main/full_compact_kl_seed42 \
  --min-count 20
```

该脚本用于分析不同 robust view 下的可靠性、attention 分布和退化影响。

---

## 13. Obfuscapk 外部真实混淆评估

内部 robust views 是 synthetic perturbations，适合机制验证，但仍属于内部模拟。

为了增强论文可信度，建议额外使用 Obfuscapk 风格的真实混淆外部评估。

实验定位：

```text
不参与训练
不参与调参
只用于最终 external robustness evaluation
使用 train split 冻结的 Manifest vocab
报告混淆成功率、构图成功率、配对成功率
```

### 13.1 构建 Obfuscapk label CSV

```bash
python scripts/build_obfuscapk_label_csvs.py \
  --config config/extract_obfuscapk.yaml \
  --clean-labels results/labels/test.csv \
  --output-dir results/labels_obfuscapk
```

当前 label CSV 推荐字段：

```text
id
sha256
label
year
split
source_id
apk_name
```

其中：

```text
id / sha256   混淆后 APK hash
source_id     原始 clean APK hash
split         Obfuscapk scenario 名称
apk_name      混淆后 APK 文件名
```

### 13.2 构建 Obfuscapk AEG PT

使用冻结 train vocab，不要从 Obfuscapk 数据重建 vocab。

```bash
python scripts/build_aeg_pts_direct.py \
  --config config/extract_obfuscapk.yaml \
  --no-rebuild-vocab \
  --resume \
  --workers 4
```

### 13.3 评估已训练 checkpoint

```bash
python scripts/evaluate_aeg_checkpoint.py \
  --checkpoint results/aeg_robust/main/full_compact_kl_seed42/best.pt \
  --config config/eval_obfuscapk.yaml \
  --output-dir results/aeg_robust/main/full_compact_kl_seed42/external
```

`config/eval_obfuscapk.yaml` 示例：

```yaml
scenarios:
  rebuild:
    pt_dir: D:/pts_obfuscapk/rebuild
    csv: results/labels_obfuscapk/rebuild.csv
    strict_integrity: false

  rename:
    pt_dir: D:/pts_obfuscapk/rename
    csv: results/labels_obfuscapk/rename.csv
    strict_integrity: false

  string_encrypt:
    pt_dir: D:/pts_obfuscapk/string_encrypt
    csv: results/labels_obfuscapk/string_encrypt.csv
    strict_integrity: false

  reflection:
    pt_dir: D:/pts_obfuscapk/reflection
    csv: results/labels_obfuscapk/reflection.csv
    strict_integrity: false

  call_indirection:
    pt_dir: D:/pts_obfuscapk/call_indirection
    csv: results/labels_obfuscapk/call_indirection.csv
    strict_integrity: false

  control_flow:
    pt_dir: D:/pts_obfuscapk/control_flow
    csv: results/labels_obfuscapk/control_flow.csv
    strict_integrity: false

  junk_code:
    pt_dir: D:/pts_obfuscapk/junk_code
    csv: results/labels_obfuscapk/junk_code.csv
    strict_integrity: false

  manifest_noise:
    pt_dir: D:/pts_obfuscapk/manifest_noise
    csv: results/labels_obfuscapk/manifest_noise.csv
    strict_integrity: false
```

`strict_integrity: false` 用于外部真实测试场景，因为 Obfuscapk 可能存在：

```text
混淆成功但标签无法映射
标签存在但 PT 构建失败
PT 存在但不在 CSV 中
```

外部评估阶段通常只评估“有标签且成功构图”的交集样本。

---

## 14. Obfuscapk clean-vs-obfuscated 配对统计

在 external evaluation 后，使用配对统计脚本：

```bash
python scripts/summarize_obfuscapk_pairs.py \
  --clean results/aeg_robust/main/full_compact_kl_seed42/diagnostics_test_clean.csv \
  --external-dir results/aeg_robust/main/full_compact_kl_seed42/external \
  --output-dir results/aeg_robust/main/full_compact_kl_seed42/external_pairs
```

该脚本按如下规则配对：

```text
clean diagnostics:
  sid = 原始 clean APK hash

external diagnostics:
  sid       = 混淆后 APK hash
  source_id = 原始 clean APK hash

join key:
  clean.sid == external.source_id
```

输出目录：

```text
external_pairs/
  paired_rebuild.csv
  paired_rename.csv
  paired_string_encrypt.csv
  paired_reflection.csv
  paired_call_indirection.csv
  paired_control_flow.csv
  paired_junk_code.csv
  paired_manifest_noise.csv
  paired_all.csv
  summary_pairs.csv
  summary_pairs.json
```

建议报告指标：

```text
external_rows
rows_with_source_id
paired_count
source_id_rate
pair_rate
flip_rate
clean_acc_on_paired
obf_acc_on_paired
acc_drop_on_paired
mean_prob_abs_delta
mean_true_confidence_drop
```

其中：

```text
flip_rate = clean 预测类别 != obfuscated 预测类别 的比例

true_confidence_drop =
  clean view 中真实类别置信度
  -
  obfuscated view 中真实类别置信度
```

### 14.1 P1：label mismatch 检查

为了保证 paired evaluation 的标签映射没有污染，配对脚本应检查：

```text
clean diagnostics 中的 label
是否等于
external diagnostics 中的 label
```

也就是：

```python
label_mismatch = int(clean_label != label)
```

并在 summary 中统计：

```text
label_mismatch_count
label_mismatch_rate
```

如果：

```text
label_mismatch_rate > 0
```

说明 `source_id` 映射或 label CSV 生成可能存在问题，这批 Obfuscapk paired 结果不应直接用于论文报告。应先修复 mapping，再重新生成 external diagnostics 和 paired summary。

建议论文使用结果前确认：

```text
source_id_rate 接近 1.0
pair_rate 足够高
label_mismatch_rate = 0
```

如果当前脚本尚未输出 `label_mismatch_count / label_mismatch_rate`，应先补齐该检查，再用于最终论文表格。

---

## 15. Obfuscapk 论文报告建议

建议至少报告：

```text
尝试混淆 APK 数量
成功混淆 APK 数量
成功构建 AEG PT 数量
进入 paired evaluation 的样本数量
benign/malware 数量
不同 scenario 的样本数量
clean Macro-F1
Obfuscapk Macro-F1
F1 absolute drop
F1 relative drop
prediction flip rate
true confidence drop
label_mismatch_rate
```

推荐表格：

| Method     | Scenario | Paired Count | Flip Rate ↓ | Clean Acc | Obf Acc | Acc Drop ↓ | True Conf. Drop ↓ | Label Mismatch |
| ---------- | -------: | -----------: | ----------: | --------: | ------: | ---------: | ----------------: | -------------: |
| CE-only    |  overall |              |             |           |         |            |                   |                |
| Plain-KL   |  overall |              |             |           |         |            |                   |                |
| Compact-KL |  overall |              |             |           |         |            |                   |                |

期望结论：

```text
Compact-KL 在真实混淆下 prediction flip rate 更低；
Compact-KL 的 paired accuracy drop 更小；
Compact-KL 的 true confidence drop 更小；
所有用于论文报告的 paired 结果 label_mismatch_rate 应为 0。
```

---

## 16. 测试与静态检查

### 16.1 Python 语法检查

```bash
python -m py_compile \
  fusion/losses.py \
  fusion/train.py \
  fusion/model.py \
  fusion/dataset.py \
  fusion/perturbations.py \
  run.py \
  scripts/build_aeg_pts_direct.py \
  scripts/evaluate_aeg_checkpoint.py \
  scripts/build_obfuscapk_label_csvs.py \
  scripts/summarize_obfuscapk_pairs.py
```

### 16.2 pytest smoke test

```bash
pytest tests/test_aeg_smoke.py
```

### 16.3 runner dry-run

```bash
python run.py final --dry-run
python run.py full_seeds --dry-run
python run.py stage1 --dry-run
python run.py stage2 --dry-run
python run.py stage3 --dry-run
python run.py all --dry-run
```

dry-run 应只出现当前路径：

```text
main/full_compact_kl_seed42.yaml
main/full_compact_kl_seed43.yaml
main/full_compact_kl_seed44.yaml
stage1/...
stage2/...
stage3/...
```

不应出现旧路径：

```text
full/ours.yaml
i1
i2
i3
multiview_contrast.yaml
no_clean_degraded_contrast.yaml
no_source_degraded_contrast.yaml
no_cross_source_contrast.yaml
```

### 16.4 Obfuscapk 配置检查

```bash
python - <<'PY'
from fusion.config_utils import load_config

extract_cfg = load_config("config/extract_obfuscapk.yaml")
eval_cfg = load_config("config/eval_obfuscapk.yaml")

print(extract_cfg["data"]["splits"])
print(eval_cfg["scenarios"].keys())
PY
```

---

## 17. 论文实验对应关系

### RQ1：源感知异构证据图是否有效，且其鲁棒性瓶颈是什么？

运行：

```bash
python run.py stage1
```

比较：

```text
AEG-only CE
API-only CE
Graph-only CE
Manifest-only
No source metadata
No relation types
No quality
No alignment
No risk
```

### RQ2：可靠性 / 冲突感知 fusion 是否缓解来源依赖？

运行：

```bash
python run.py stage2
```

比较：

```text
Latent content only
Reliability only
Conflict only
Source only
No reliability / no conflict / no source-score bias
Full fusion CE
```

### RQ3：可靠性感知 KL 一致性是否进一步提升鲁棒性？

运行：

```bash
python run.py stage3
```

比较：

```text
Full fusion CE
Full fusion Plain-KL
Full fusion Compact-KL
Compact-KL weight sweep
```

重点分析：

```text
manifest_noisy
manifest_shuffled
manifest_noisy_blind
manifest_shuffled_blind
manifest_missing
```

### RQ4：完整方法在内部 synthetic robust views 下是否稳定？

使用 full method 的 robust-test 指标，报告：

```text
clean
api_degraded
graph_degraded
api_graph_degraded
manifest_noisy
manifest_shuffled
api_missing
graph_missing
manifest_missing
all_degraded
```

### RQ5：完整方法在真实混淆 Obfuscapk 外部测试下是否泛化？

使用：

```bash
python scripts/evaluate_aeg_checkpoint.py ...
python scripts/summarize_obfuscapk_pairs.py ...
```

报告：

```text
external Macro-F1
paired flip rate
paired accuracy drop
true confidence drop
label mismatch rate
```

---

## 18. 复现实验建议

建议固定：

```text
eval.seed = 2026
```

三种 train seed：

```text
train.seed = 42
train.seed = 43
train.seed = 44
```

推荐报告方式：

```text
完整主方法：3 seeds mean ± std
大规模消融：single seed 筛选
核心 baseline：可补 3 seeds 验证
Obfuscapk external evaluation：至少对 full method、CE-only、Plain-KL、Compact-KL 做对比
```

最低论文可用实验集合：

```text
Full compact-KL, 3 seeds
AEG-only CE
Full fusion CE
Plain-KL
Compact-KL
API-only
Graph-only
Manifest-only
No source metadata
No relation types
No quality
No alignment
No risk
No reliability bias
No conflict bias
No source bias
Obfuscapk external evaluation
Obfuscapk paired flip-rate summary
```

---

## 19. 注意事项

1. Obfuscapk 样本只用于最终外部测试，不进入 train/val。
2. Obfuscapk 构建 PT 时必须使用 train split 冻结的 Manifest vocab。
3. 不要只报告成功样本数量，要同时报告失败和过滤情况。
4. paired evaluation 前必须检查 `source_id_rate`、`pair_rate` 和 `label_mismatch_rate`。
5. 如果 `label_mismatch_rate > 0`，不要把 paired flip-rate 结果写进论文主表，应先修复 label mapping。
6. 当前方法不要再写成 contrastive learning，应统一写成 reliability-aware multi-view KL consistency learning。
7. Graph-only 仍受敏感 API 中心子图选择影响，不得宣称为 API-independent baseline。
8. `code_manifest_conflict` 是语义分歧代理；没有消融和盲扰动证据时，不得宣称模型识别了真实逻辑冲突。
