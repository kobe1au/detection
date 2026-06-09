# 代码审查报告：实际存在的问题总结

**审查日期:** 2026-06-09  
**审查范围:** fusion/model.py, fusion/dataset.py, fusion/perturbations.py, run.py, config/base.yaml

---

## 📊 审查总结

**总体代码质量:** 4.2/5.0  
**主要问题数量:** 7 个  
**次要问题数量:** 5 个  
**严重程度分布:** 🔴 高 (2) | 🟡 中 (5) | 🟢 低 (5)

---

## 🔴 P0 - 高优先级问题（影响实验可信度）

### 1. ❌ node_input_dim 自动 pad/truncate（model.py:262-270）

**问题:** 训练/验证/测试时，如果 PT 文件的 `node_x` 维度不匹配，会**静默 pad 或 truncate**

```python
# fusion/model.py:262-270 (当前代码)
def _initial_node_state(self, data: Batch) -> torch.Tensor:
    x = data.x.float()
    if x.size(1) != self.input_proj.in_features:
        if x.size(1) < self.input_proj.in_features:
            pad = x.new_zeros((x.size(0), self.input_proj.in_features - x.size(1)))
            x = torch.cat([x, pad], dim=-1)  # ❌ 静默 pad
        else:
            x = x[:, : self.input_proj.in_features]  # ❌ 静默 truncate
```

**影响:**
- PT 文件维度不一致时不会报错
- 可能使用了错误的 PT 文件而不知道
- 实验结果不可解释

**建议修复:**
```python
def _initial_node_state(self, data: Batch) -> torch.Tensor:
    x = data.x.float()
    expected_dim = self.input_proj.in_features
    
    if x.size(1) != expected_dim:
        raise RuntimeError(
            f"node_x dimension mismatch!\n"
            f"Expected: {expected_dim}, Got: {x.size(1)}\n"
            f"Regenerate PT files or use matching extraction config.\n"
            f"Mode: {'training' if self.training else 'evaluation'}"
        )
    
    # ... rest unchanged
```

**优先级:** 🔴 最高  
**工作量:** 30 分钟

---

### 2. ⚠️ code_rel 可能产生 NaN/Inf（model.py:361-365）

**问题:** `code_rel` 计算在 `r_api` 或 `r_graph` 为 0 时，`sqrt(0 * 0)` 可能导致数值不稳定

```python
# fusion/model.py:361-365 (当前代码)
code_rel = (
    (r_api * r_graph).sqrt()  # ❌ r_api=0 或 r_graph=0 时可能不稳定
    * (0.5 + 0.5 * q_align.clamp(0.0, 1.0))
).clamp(0.0, 1.0)
```

**问题场景:**
- `r_api = 0` 或 `r_graph = 0` 时，语义上代码可靠性应为 0
- 但 `sqrt(0)` 的梯度可能导致数值问题
- 当前代码虽然不会产生 NaN，但语义不够清晰

**建议修复:**
```python
# 保留零语义，但防止数值问题
EPS = 1e-12
product = r_api.clamp_min(0.0) * r_graph.clamp_min(0.0)

code_rel_nonzero = (
    product.clamp_min(EPS).sqrt()
    * (0.5 + 0.5 * q_align.clamp(0.0, 1.0))
)

# 关键: 只在 product > 0 时使用计算值，否则强制为 0
code_rel = torch.where(
    product > 0,
    code_rel_nonzero,
    torch.zeros_like(code_rel_nonzero)
).clamp(0.0, 1.0)
```

**优先级:** 🟡 中  
**工作量:** 1 小时

---

## 🟡 P1 - 中优先级问题（工程质量）

### 3. ⚠️ eval.seed 与 train.seed 不同（base.yaml:14,89）

**问题:** 当前配置文件中 `train.seed: 42` 和 `eval.seed: 2026` **不同**

```yaml
# config/experiments/aeg_robust/base.yaml
train:
  seed: 42  # 训练种子

eval:
  seed: 2026  # ❌ 与训练不同
```

**实际情况:**
- 你说得对，单次实验应该用**同一个 seed**
- 多种子验证应该跑 3-5 个**完整实验**，每个实验内部 seed 一致

**正确做法:**
```yaml
train:
  seed: 42
eval:
  seed: 42  # ✅ 应该相同，保证可复现性
```

**多种子验证:**
```yaml
# full_seeds/seed_42.yaml
train: {seed: 42}
eval: {seed: 42}

# full_seeds/seed_2026.yaml
train: {seed: 2026}
eval: {seed: 2026}

# full_seeds/seed_12345.yaml
train: {seed: 12345}
eval: {seed: 12345}
```

**优先级:** 🟢 低（但需要修正）  
**工作量:** 5 分钟

---

### 4. ⚠️ manifest_shuffled 单样本检查缺失（dataset.py:315-318）

**问题:** 单样本 split 无法提供 donor，代码会静默使用 zeroed Manifest，但没有提前检查

```python
# fusion/dataset.py:315-318
if donor is None:
    # A one-sample split cannot supply a donor; use zeroed Manifest
    # evidence instead of silently keeping the original.
    _zero_manifest_nodes(aug)
```

**影响:**
- 单样本 split 会静默执行 `_zero_manifest_nodes`
- 用户不知道 manifest_shuffled 实际上变成了 manifest_missing
- 实验结果可能误导

**建议修复:**
```python
# 在 AEGDataset.__init__ 添加检查
if "manifest_shuffled" in self.aug_views and len(self.samples) <= 1:
    raise AEGDatasetConfigError(
        f"manifest_shuffled requires at least 2 samples, "
        f"but {self.pt_dir} has only {len(self.samples)} sample(s). "
        f"Remove manifest_shuffled from aug_views or use a larger split."
    )
```

**优先级:** 🟢 低  
**工作量:** 30 分钟

---

### 5. 📌 Magic numbers 未常量化

**问题:** 代码中多处使用 magic numbers，可维护性差

**示例:**
```python
# fusion/model.py:362
code_rel = (
    (r_api * r_graph).sqrt()
    * (0.5 + 0.5 * q_align.clamp(0.0, 1.0))  # ❌ 0.5 是什么？
).clamp(0.0, 1.0)

# fusion/model.py:400
code_emb = 0.5 * (method_emb + api_family_emb)  # ❌ 0.5 权重
manifest_emb = 0.5 * (permission_emb + component_emb)

# fusion/perturbations.py:各处
# 0.4, 0.6, 0.8, 0.9 等 cf_weight
```

**建议修复:**
```python
# fusion/constants.py (新增)
# Code reliability modulation factors
Q_ALIGN_BASE_WEIGHT = 0.5  # Baseline code reliability without alignment
Q_ALIGN_MODULATION = 0.5   # Alignment boost factor

# Token fusion weights
CODE_EMB_METHOD_WEIGHT = 0.5
CODE_EMB_API_WEIGHT = 0.5

# Contrastive weights per view
CF_WEIGHTS = {
    "api_degraded": 0.4,
    "graph_degraded": 0.4,
    "manifest_noisy": 0.8,
    "manifest_shuffled": 0.9,
    "all_degraded": 1.0,
}
```

**优先级:** 🟢 低  
**工作量:** 2 小时

---

### 6. ⚠️ 硬编码路径（base.yaml:5,8,11）

**问题:** PT 目录使用绝对路径 `D:/pts_aeg/`，不可移植

```yaml
# config/experiments/aeg_robust/base.yaml
data:
  train:
    pt_dir: D:/pts_aeg/train  # ❌ 硬编码绝对路径
  val:
    pt_dir: D:/pts_aeg/val
  test:
    pt_dir: D:/pts_aeg/test
```

**建议修复:**
```yaml
data:
  train:
    pt_dir: ${PT_ROOT}/train  # 环境变量
  val:
    pt_dir: ${PT_ROOT}/val
  test:
    pt_dir: ${PT_ROOT}/test

# 或使用相对路径
data:
  train:
    pt_dir: data/pt/aeg/train
```

**优先级:** 🟢 低  
**工作量:** 1 小时

---

### 7. ⚠️ 缺少 PT 验证脚本

**问题:** 没有批量验证 PT 文件结构的脚本

**建议新增:** `scripts/verify_pt_structure.py`

```python
def verify_pt_structure(pt_path: Path, expected_config: dict) -> bool:
    """验证 PT 文件结构是否符合预期"""
    try:
        data = torch.load(pt_path, map_location="cpu")
        
        # 检查必需字段
        assert "graph" in data
        assert "label" in data
        
        graph = data["graph"]
        
        # 检查节点特征维度
        expected_node_dim = expected_config.get("node_feature_dim", 128)
        actual_node_dim = graph.x.size(1) if hasattr(graph, 'x') else 0
        
        if actual_node_dim != expected_node_dim:
            print(f"❌ {pt_path.name}: node_dim={actual_node_dim}, expected={expected_node_dim}")
            return False
        
        print(f"✅ {pt_path.name}: valid")
        return True
        
    except Exception as e:
        print(f"❌ {pt_path.name}: {e}")
        return False
```

**优先级:** 🟡 中  
**工作量:** 半天

---

## 🟢 P2 - 低优先级问题（优化建议）

### 8. 📊 缺少 blind eval 的显式配置

**问题:** `base.yaml` 中 `robust_views` 已包含 `manifest_noisy_blind` 和 `manifest_shuffled_blind`，但没有单独的消融配置

**建议:** 这个其实已经配置好了，不是问题。`base.yaml:110-111` 已有 blind 模式。

---

### 9. 📊 缺少 quality shortcut 消融配置

**问题:** 没有 `i1/no_node_edge_quality.yaml` 等消融配置

**建议新增:**
```yaml
# config/experiments/aeg_robust/i1/no_node_edge_quality.yaml
base: ../base.yaml
model:
  use_node_quality: false
  use_edge_quality: false

# config/experiments/aeg_robust/i1/shuffled_quality.yaml
base: ../base.yaml
# 需要在 dataset/model 中实现 shuffle 逻辑
```

**优先级:** 🟡 中  
**工作量:** 2-3 天（含实现）

---

### 10. 📝 缺少 attention 可视化脚本

**问题:** 没有验证创新点 3 (Manifest shortcut suppression) 的可视化脚本

**建议新增:** `scripts/visualize_attention_manifest_suppression.py`

**优先级:** 🟡 中  
**工作量:** 1 天

---

### 11. ⚠️ deterministic_aug 默认 False（dataset.py:212）

**问题:** `deterministic_aug: bool = False` 导致多次运行同一实验，增强结果不同

```python
# fusion/dataset.py:212
def __init__(
    self,
    # ...
    deterministic_aug: bool = False,  # ❌ 默认非确定性
):
```

**影响:**
- 同一 seed 多次运行，增强视图可能不同
- 可复现性降低

**建议:**
- 论文实验应该默认 `deterministic_aug: True`
- 或在 `base.yaml` 中显式设置

**优先级:** 🟢 低  
**工作量:** 10 分钟

---

### 12. 📊 RobustVal 权重配置不清晰（base.yaml:87-102）

**问题:** `robust_val` 各 scenario 的权重含义不明确

```yaml
robust_val:
  clean_weight: 0.5
  scenarios:
    - name: api_graph_degraded
      weight: 0.4  # ❌ 这些权重如何组合？
    - name: manifest_noisy
      weight: 0.3
```

**建议:** 添加注释说明权重计算公式

---

## ✅ 代码中做得好的地方

1. ✅ **blind 模式语义正确**（perturbations.py:280-290）
   - blind 模式不更新 `pert_manifest`
   - 语义清晰，设计合理

2. ✅ **payload contract 验证完整**（payload_contract.py）
   - 严格的 PT 文件格式验证
   - 错误信息清晰

3. ✅ **消融实验组织清晰**（run.py:25-85）
   - i1/i2/i3 分组合理
   - 配置文件结构良好

4. ✅ **perturbation 语义明确**（perturbations.py）
   - 各种扰动模式实现完整
   - `pert_*` 和 `q_*` 更新一致

5. ✅ **donor 模式设计合理**（dataset.py:320-340）
   - cyclic 和 opposite_label 两种模式
   - 单样本情况有 fallback

---

## 📋 修复优先级总结

### 立即修复（P0，1 天）
1. 🔴 **node_input_dim 严格检查** (30 分钟)
2. 🟡 **code_rel 数值稳定** (1 小时)
3. 🟢 **eval.seed 改为与 train.seed 相同** (5 分钟)

### 短期修复（P1，1-2 周）
4. 🟡 **PT 验证脚本** (半天)
5. 🟡 **quality shortcut 消融** (2-3 天)
6. 🟡 **attention 可视化** (1 天)
7. 🟢 **单样本 split 检查** (30 分钟)

### 长期优化（P2，可选）
8. 🟢 **Magic numbers 常量化** (2 小时)
9. 🟢 **硬编码路径环境变量** (1 小时)
10. 🟢 **deterministic_aug 默认值** (10 分钟)

---

## 🎯 最终结论

**代码整体质量很高，主要问题是：**

1. ❌ **node_input_dim 自动 pad/truncate** - 必须修复
2. ⚠️ **eval.seed 配置错误** - 需要修正（但你已经指出了）
3. ⚠️ **缺少 quality shortcut 消融实验** - 需要补充
4. ⚠️ **缺少 PT 验证脚本** - 建议添加

**不是问题的部分：**
- ✅ blind 模式实现正确
- ✅ perturbation 语义清晰
- ✅ 实验配置组织合理
- ✅ payload contract 验证完整

**总工作量估算:** 
- P0 修复: 1-2 天
- P1 补充: 1-2 周
- P2 优化: 可选
