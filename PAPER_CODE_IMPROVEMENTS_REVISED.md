# 论文代码改进计划 - 修正版

**基于实际需求的优先级排序**  
**创建时间:** 2026-06-08  
**修正原因:** 原计划优先级判断有误,遗漏关键问题

---

## 🎯 改进原则

### 核心目标
1. **论文质量优先** - 确保实验论证完整
2. **数据完整性优先** - 避免 silent fail
3. **实验公平性优先** - 保证可重现性
4. **工程优化靠后** - 有 benchmark 才做

### 判断标准
- **P0**: 影响论文论证或数据正确性
- **P1**: 支撑论文 ablation 或可读性
- **P2**: 工程便利性,不影响当前实验

---

## 🔴 P0: 核心问题修复 (必须完成,1周)

### 1. Blind Manifest Corruption (最重要!) ✨

**问题:**
创新点3声称"冲突感知融合可以自主检测 code-manifest 不一致",但当前实验中:

```python
# 当前 manifest_shuffled
target.pert_manifest = torch.tensor([1.0])  # ❌ 模型被明确告知 manifest 损坏
```

模型通过 `pert_manifest` 信号就知道 manifest 不可信,没有真正"自主检测冲突"。

**需要添加:**
```python
# Blind variant: 模型不知道 manifest 被篡改
- manifest_shuffled_blind  # pert_manifest 保持原值
- manifest_noisy_blind     # pert_manifest 保持原值
```

**实现位置:** `fusion/perturbations.py`

**修改方案:**

```python
def apply_aeg_view(data: Data, *, view: str, strength: float = 0.5) -> Data:
    # ... 现有代码 ...
    
    elif view == "manifest_shuffled_blind":
        # ✅ 不修改 pert_manifest,让模型自主检测
        out.cf_weight = torch.tensor([0.9], dtype=torch.float32)
        # pert_manifest 保持原值不变
    
    elif view == "manifest_noisy_blind":
        _degrade_manifest(out, strength, noisy=True, blind=True)
        # blind=True 时不设置 pert_manifest
        out.cf_weight = torch.tensor([0.8 * strength], dtype=torch.float32)
```

**在 _degrade_manifest 中:**

```python
def _degrade_manifest(data: Data, strength: float, *, missing: bool = False, noisy: bool = False, blind: bool = False) -> None:
    # ... 现有降级逻辑 ...
    
    if not blind:  # ✅ 只有非 blind 模式才设置 pert_manifest
        _set_scalar(data, "pert_manifest", 1.0 if missing else max(...))
    
    refresh_apk_node_quality(data)
    refresh_risk_node_quality(data)
```

**验证实验:**

```yaml
# config/experiments/aeg_robust/ablation/blind_corruption.yaml
robust:
  train_views:
    - manifest_shuffled_blind   # 新增
    - manifest_noisy_blind       # 新增

eval:
  robust_views:
    - manifest_shuffled_blind
    - manifest_noisy_blind
```

**预期效果:**
- 在 blind 场景下,模型性能应该主要依赖 conflict detection (创新点3)
- 如果 blind 下性能暴跌,说明模型过度依赖 pert_manifest 信号
- 论文可以写: "即使在 blind corruption 下,模型仍保持 XX% F1,验证了冲突感知融合的有效性"

**优先级:** 🔴🔴🔴 **最高** - 影响创新点3的论证逻辑

**工时:** 3 小时

---

### 2. eval.seed 独立于 train.seed

**问题:**

当前代码中 (`train.py`):

```python
def _make_dataset(cfg, split, *, aug=False, view=None, strength=None):
    train_seed = int((cfg.get("train", {}) or {}).get("seed", 42))
    eval_seed = int((cfg.get("eval", {}) or {}).get("seed", train_seed + 100_000))
    dataset_seed = eval_seed if view else train_seed  # ❌ 评估扰动仍然用 eval_seed
```

但 `_loader` 中:

```python
def _loader(cfg, dataset, *, train: bool):
    train_seed = int(train_cfg.get("seed", 42))
    eval_seed = int((cfg.get("eval", {}) or {}).get("seed", train_seed + 100_000))
    generator.manual_seed(train_seed if train else eval_seed)  # ❌ eval loader 用 eval_seed
```

**问题:**
- 多 seed 实验 (42, 52, 62) 时,如果 eval_seed 派生自 train_seed,三次实验的评估扰动可能高度相关
- 影响实验公平性和可重现性

**修改方案:**

```python
# train.py
def _make_dataset(cfg, split, *, aug=False, view=None, strength=None):
    train_cfg = cfg.get("train", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    
    train_seed = int(train_cfg.get("seed", 42))
    # ✅ eval_seed 完全独立,不派生
    eval_seed = int(eval_cfg.get("seed", 2026))
    
    # ✅ 评估扰动确定性取决于 view 参数
    if view:  # deterministic aug for evaluation
        dataset_seed = eval_seed
    else:
        dataset_seed = train_seed
    
    return AEGDataset(..., seed=dataset_seed, ...)

def _loader(cfg, dataset, *, train: bool):
    train_cfg = cfg.get("train", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    
    train_seed = int(train_cfg.get("seed", 42))
    eval_seed = int(eval_cfg.get("seed", 2026))  # ✅ 独立
    
    generator = torch.Generator()
    generator.manual_seed(train_seed if train else eval_seed)
    return DataLoader(..., generator=generator, ...)
```

**配置修改:**

```yaml
# config/experiments/aeg_robust/base.yaml
train:
  seed: 42

eval:
  seed: 2026  # ✅ 完全独立的 seed
```

**验证:**

```bash
# 测试不同 train_seed 下 eval 结果的独立性
python run.py full_seeds

# 检查三个 seed 的 val metrics 是否合理
```

**优先级:** 🔴 **P0** - 影响实验公平性

**工时:** 1 小时

---

### 3. Manifest Vocab Fingerprint 校验

**问题:**

当前 `build_aeg_pts_direct.py` 中有 vocab metadata 校验 (行 790-808),但只在**生成 PT 时**校验。

训练时加载 vocab 没有校验当前 train CSV 是否匹配:

```python
# train.py 或 dataset.py 加载 vocab 时
vocab = load_yaml("config/manifest_vocab_aeg.yaml")
# ❌ 没有检查 vocab 是否是从当前 train.csv 生成的
```

**风险:**
- 如果用了旧 vocab 训练新数据,特征向量可能错位
- Silent fail,难以排查

**修改方案:**

在 `fusion/manifest_features.py` 中添加:

```python
def validate_vocab_for_train_csv(
    vocab: dict,
    train_csv_path: Path,
    *,
    strict: bool = True
) -> None:
    """验证 vocab 是否与当前 train CSV 匹配"""
    metadata = vocab.get("metadata") or {}
    
    # 读取当前 train CSV 的 IDs
    current_ids = set()
    with train_csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("sha256") or row.get("id", "")).strip().lower()
            if sid:
                current_ids.add(sid)
    
    current_count = len(current_ids)
    current_fingerprint = hashlib.sha256(
        ";".join(sorted(current_ids)).encode("utf-8")
    ).hexdigest()
    
    # 对比 metadata
    expected_count = int(metadata.get("source_id_count", -1))
    expected_fp = str(metadata.get("source_id_fingerprint", ""))
    
    if strict:
        if expected_count != current_count:
            raise ValueError(
                f"Vocab was built from {expected_count} train samples, "
                f"but current train CSV has {current_count} samples. "
                f"Rebuild vocab with --rebuild-vocab."
            )
        if expected_fp != current_fingerprint:
            raise ValueError(
                f"Vocab source fingerprint mismatch. "
                f"Expected: {expected_fp[:16]}..., "
                f"Got: {current_fingerprint[:16]}... "
                f"Rebuild vocab from current train CSV."
            )
    else:
        # Warn only
        if expected_count != current_count or expected_fp != current_fingerprint:
            import warnings
            warnings.warn(
                f"Vocab may not match current train CSV. "
                f"Consider rebuilding with --rebuild-vocab."
            )
```

**在训练入口调用:**

```python
# train.py 的 run() 函数中
def run(cfg: dict[str, Any]) -> dict[str, Any]:
    # ... 现有代码 ...
    
    # ✅ 验证 vocab
    manifest_cfg = cfg.get("manifest", {}) or {}
    vocab_path = manifest_cfg.get("vocab_path", "config/manifest_vocab_aeg.yaml")
    if Path(vocab_path).exists():
        vocab = load_yaml(vocab_path)
        train_csv = data_cfg.get("train", {}).get("csv")
        if train_csv and Path(train_csv).exists():
            validate_vocab_for_train_csv(
                vocab, 
                Path(train_csv),
                strict=bool(data_cfg.get("strict_vocab_check", True))
            )
    
    # 继续训练...
```

**优先级:** 🔴 **P0** - 数据完整性

**工时:** 2 小时

---

### 4. node_input_dim Strict 检查

**问题:**

当前 `model.py:276-283`:

```python
def _initial_node_state(self, data: Batch) -> torch.Tensor:
    x = data.x.float()
    if x.size(1) != self.input_proj.in_features:
        if x.size(1) < self.input_proj.in_features:
            pad = x.new_zeros((x.size(0), self.input_proj.in_features - x.size(1)))
            x = torch.cat([x, pad], dim=-1)  # ❌ 静默 pad
        else:
            x = x[:, : self.input_proj.in_features]  # ❌ 静默 truncate
```

**问题:**
- 与 strict payload contract 理念不一致
- 维度不匹配应该是配置错误,不应该静默修复
- 可能掩盖 PT 文件版本不匹配的问题

**修改方案:**

```python
def _initial_node_state(self, data: Batch) -> torch.Tensor:
    x = data.x.float()
    expected_dim = self.input_proj.in_features
    actual_dim = x.size(1)
    
    if actual_dim != expected_dim:
        # ✅ 默认报错,只有明确配置 auto_adjust 才容忍
        if not getattr(self, "auto_adjust_input_dim", False):
            raise RuntimeError(
                f"Node feature dimension mismatch: "
                f"model expects {expected_dim}, "
                f"but PT file has {actual_dim}. "
                f"Check aeg.node_feature_dim in extraction config "
                f"matches model.node_input_dim in training config. "
                f"Set model.auto_adjust_input_dim=true to allow auto padding/truncate."
            )
        
        # Auto adjust (仅在配置明确允许时)
        if actual_dim < expected_dim:
            pad = x.new_zeros((x.size(0), expected_dim - actual_dim))
            x = torch.cat([x, pad], dim=-1)
        else:
            x = x[:, :expected_dim]
    
    # ... 其余代码 ...
```

**配置添加:**

```yaml
# config/experiments/aeg_robust/base.yaml
model:
  node_input_dim: 128
  auto_adjust_input_dim: false  # ✅ 默认 false,严格检查
```

**优先级:** 🔴 **P0** - 数据完整性

**工时:** 1 小时

---

### 5. Resume Precheck 打印

**问题:**

当前 resume 机制在后台静默检查,用户不知道有多少 PT 可以 resume,多少需要重新生成。

**修改方案:**

在 `build_aeg_pts_direct.py` 的主流程中:

```python
def main():
    # ... 现有扫描逻辑 ...
    
    # ✅ Precheck: 统计可 resume 的 PT 文件
    if cfg["resume"]:
        print("\n" + "=" * 80)
        print("Resume Precheck")
        print("=" * 80)
        
        resumable = 0
        pending = 0
        
        for job in jobs:
            out_path = _out_path(job, cfg)
            resume_row = _resume_existing(job, cfg)
            if resume_row is not None:
                resumable += 1
            else:
                pending += 1
        
        print(f"\n✅ Resumable PT files: {resumable}/{len(jobs)}")
        print(f"⏳ Pending generation: {pending}/{len(jobs)}")
        
        if resumable == len(jobs):
            print("\n🎉 All PT files can be resumed, generation will complete in seconds!")
        elif resumable > 0:
            print(f"\n⚡ {resumable} PT files will be reused, saving ~{resumable * 30 / 3600:.1f} hours")
        
        print("=" * 80 + "\n")
    
    # 继续正常生成流程...
```

**效果:**

```
================================================================================
Resume Precheck
================================================================================

✅ Resumable PT files: 10334/15000
⏳ Pending generation: 4666/15000

⚡ 10334 PT files will be reused, saving ~86.1 hours
================================================================================
```

**优先级:** 🔴 **P0** - 用户体验

**工时:** 1 小时

---

## 🟡 P1: 论文支撑实验 (2周)

### 6. Quality-Only Classifier Ablation

**目的:** 证明图结构和融合机制的必要性

**实验配置:**

```yaml
# config/experiments/aeg_robust/ablation/quality_only.yaml
base: ../base.yaml

train:
  output_dir: results/aeg_robust/ablation/quality_only

model:
  # 使用简单分类器,只基于质量标量
  classifier_input: quality_only  # 新增参数
```

**实现:**

```python
# model.py
class AEGModel(nn.Module):
    def __init__(self, ..., classifier_input="full"):
        self.classifier_input = classifier_input
        
        if classifier_input == "quality_only":
            # 只用质量标量预测
            self.classifier = nn.Sequential(
                nn.Linear(4, 64),  # [q_api, q_graph, q_manifest, q_align]
                nn.ReLU(),
                nn.Linear(64, num_classes),
            )
    
    def forward(self, data):
        # ... 图编码和融合 ...
        
        if self.classifier_input == "quality_only":
            quality_vec = torch.stack([q_api, q_graph, q_manifest, q_align], dim=1)
            logits = self.classifier(quality_vec)
        else:
            logits = self.classifier(fused)
        
        return logits, extra
```

**预期:** Quality-only 应该显著差于 full 方法

**工时:** 4 小时

---

### 7. Reliability-Weighted vs Unweighted Contrast

**目的:** 验证创新点2中可靠性加权的必要性

**当前已有:**

```yaml
# i2/unweighted_contrast.yaml
loss:
  reliability_weighted_contrast: false
```

**补充实验:** 加入 per-source 可靠性加权的消融

```yaml
# config/experiments/aeg_robust/ablation/contrast_variants.yaml

# Variant 1: 无可靠性加权
loss:
  reliability_weighted_contrast: false

# Variant 2: 仅 fused-level 加权
loss:
  reliability_weighted_contrast: true
  reliability_weighted_source: false  # 新增

# Variant 3: Full (当前)
loss:
  reliability_weighted_contrast: true
  reliability_weighted_source: true
```

**工时:** 3 小时 (实验运行)

---

### 8. Augmented CE Baseline

**当前已有:**

```yaml
# i2/augmented_ce_only.yaml
loss:
  clean_degraded_contrast_weight: 0.0
  source_degraded_contrast_weight: 0.0
  cross_source_contrast_weight: 0.0
```

**补充:** 确保这个实验在 full seeds 下跑过

**工时:** 实验已配置,只需运行

---

### 9. Magic Number 常量化

**目的:** 提高代码可读性,方便论文写作时引用

**实现:**

在 `model.py` 顶部添加:

```python
# Fusion token configuration
@dataclass(frozen=True)
class FusionTokenConfig:
    """Configuration for multi-source token fusion."""
    
    # Token composition weights
    CODE_METHOD_WEIGHT: float = 0.5
    CODE_API_WEIGHT: float = 0.5
    MANIFEST_PERMISSION_WEIGHT: float = 0.5
    MANIFEST_COMPONENT_WEIGHT: float = 0.5
    
    # Token names (for logging and visualization)
    TOKEN_NAMES: tuple[str, ...] = (
        "method",
        "api_family",
        "permission",
        "component",
        "risk",
        "string_hint",
        "global",
    )
    
    # Conflict sensitivity per token
    # Higher = more suppressed when code-manifest conflict is detected
    TOKEN_CONFLICT_SENSITIVITY: tuple[float, ...] = (
        0.0,   # method: code evidence, conflict-insensitive
        0.0,   # api_family: code evidence, conflict-insensitive
        1.0,   # permission: manifest evidence, fully sensitive
        1.0,   # component: manifest evidence, fully sensitive
        0.5,   # risk: mixed evidence, partially sensitive
        0.0,   # string_hint: code-derived, conflict-insensitive
        0.25,  # global: aggregated, mildly sensitive
    )
    
    # Source type IDs per token
    TOKEN_SOURCE_IDS: tuple[int, ...] = (
        SOURCE_TYPES["code"],      # method
        SOURCE_TYPES["code"],      # api_family
        SOURCE_TYPES["manifest"],  # permission
        SOURCE_TYPES["manifest"],  # component
        SOURCE_TYPES["derived"],   # risk
        SOURCE_TYPES["derived"],   # string_hint
        SOURCE_TYPES["derived"],   # global
    )


# Global instance
FUSION_CONFIG = FusionTokenConfig()
```

**使用:**

```python
# model.py 中替换硬编码
code_emb = (
    FUSION_CONFIG.CODE_METHOD_WEIGHT * method_emb 
    + FUSION_CONFIG.CODE_API_WEIGHT * api_family_emb
)

token_conflict_sensitivity = torch.tensor(
    FUSION_CONFIG.TOKEN_CONFLICT_SENSITIVITY,
    device=data.x.device,
)
```

**优先级:** 🟡 **P1** - 代码可读性

**工时:** 2 小时

---

## 🟢 P2: 工程优化 (可选)

### 10. Path Resolve 支持环境变量

**实现:** 见原计划,但优先级降低到 P2

**工时:** 4 小时

---

### 11. Manifest Shuffle 单样本检查

**实现:** 见原计划

**工时:** 1 小时

---

### 12. Attention 可视化 (围绕鲁棒场景)

**重点:** 不是画静态分布,而是:

```python
# scripts/visualize_attention_robust.py

# 1. Clean vs Blind Corruption
plot_attention_comparison(
    clean_csv="diagnostics_test_clean.csv",
    corrupt_csv="diagnostics_test_manifest_shuffled_blind.csv",
    title="Attention Shift under Blind Manifest Corruption"
)

# 2. Conflict-Sensitive Token Analysis
plot_token_attention_vs_conflict(
    csv="diagnostics_test_clean.csv",
    tokens=["permission", "component"],  # High conflict sensitivity
)

# 3. Strength Effect
plot_attention_vs_strength(
    views=["manifest_degraded@0.1", "manifest_degraded@0.3", "manifest_degraded@0.5"],
    token="permission",
)
```

**工时:** 6 小时

---

### 13. Temperature Sweep (3 点即可)

**实验配置:**

```yaml
# config/experiments/aeg_robust/ablation/temperature_sweep.yaml

# Only test 3 values, not 6
- temperature: 0.1
- temperature: 0.2  # current
- temperature: 0.5
```

**工时:** 2 小时 (配置) + 实验运行

---

### 14. PT Cache Benchmark (先测试)

**不要默认启用!**

**Benchmark 脚本:**

```python
# scripts/benchmark_pt_cache.py

import time
from fusion.dataset import AEGDataset
from torch.utils.data import DataLoader

# Test 1: No cache
ds = AEGDataset(..., use_cache=False)
loader = DataLoader(ds, batch_size=24, num_workers=4)

start = time.time()
for batch in loader:
    pass
no_cache_time = time.time() - start

# Test 2: With cache
ds = AEGDataset(..., use_cache=True, cache_size=512)
# ... same

cache_speedup = no_cache_time / cache_time
print(f"Speedup: {cache_speedup:.2f}x")
```

**只有 speedup > 1.3x 才考虑启用**

**工时:** 4 小时

---

## 📅 实施时间表

### Week 1: P0 核心修复

| 任务 | 工时 | 优先级 |
|------|------|--------|
| Blind corruption | 3h | 🔴🔴🔴 |
| eval.seed 独立 | 1h | 🔴 |
| Vocab fingerprint | 2h | 🔴 |
| node_input_dim strict | 1h | 🔴 |
| Resume precheck | 1h | 🔴 |

**总计:** 8 小时

---

### Week 2-3: P1 论文支撑

| 任务 | 工时 | 优先级 |
|------|------|--------|
| Blind corruption 实验 | 8h | 🟡 |
| Quality-only ablation | 4h | 🟡 |
| Contrast variants | 3h | 🟡 |
| Magic number 常量化 | 2h | 🟡 |

**总计:** 17 小时

---

### Week 4+: P2 可选优化

| 任务 | 工时 | 优先级 |
|------|------|--------|
| Path resolve | 4h | 🟢 |
| Attention viz | 6h | 🟢 |
| Temperature sweep | 2h | 🟢 |
| PT cache benchmark | 4h | 🟢 |

**总计:** 16 小时

---

## ✅ 验收标准

### P0 完成标准

- [ ] Blind corruption 实现并测试通过
- [ ] eval.seed 独立,多 seed 实验结果合理
- [ ] Vocab fingerprint 检查在训练入口生效
- [ ] node_input_dim 不匹配时报错(非静默)
- [ ] Resume 时打印 resumable/pending 统计

### P1 完成标准

- [ ] Blind corruption F1 下降 < 10% (vs non-blind)
- [ ] Quality-only baseline 显著差于 full (> 5% F1 gap)
- [ ] Reliability-weighted contrast 优于 unweighted
- [ ] 所有 magic number 提取为常量

### P2 完成标准

- [ ] Path resolve 支持环境变量和相对路径
- [ ] Attention 可视化包含至少 3 种对比图
- [ ] Temperature sweep 完成 3 个点
- [ ] PT cache benchmark 结果记录

---

## 🎯 最重要的改进优先级

### 如果只做 3 件事:

1. **Blind corruption** ← 影响创新点3的核心论证
2. **eval.seed 独立** ← 影响实验公平性
3. **Vocab fingerprint** ← 避免 silent fail

### 如果只做 5 件事:

加上:
4. **node_input_dim strict**
5. **Quality-only ablation**

---

## 📝 与原计划的对比

| 改进项 | 原优先级 | 新优先级 | 变化原因 |
|--------|---------|---------|---------|
| 路径硬编码 | P0 | P2 | 不影响单机实验 |
| 数值稳定性 | P0 | P1 | 改法会变语义 |
| Manifest shuffle 检查 | P0 | P2 | 边缘情况 |
| PT 缓存 | P1 | P2 | 需要 benchmark |
| 边质量传播 | P1 | 不做 | 会改变模型 |
| Magic number | P1 | P1 | 保持 |
| **Blind corruption** | **缺失** | **P0** | **最重要!** |
| **eval.seed 独立** | **缺失** | **P0** | **新增** |
| **Vocab fingerprint** | **缺失** | **P0** | **新增** |

---

## 🎓 总结

### 核心改进理念

**原计划问题:**
- 优先级按"代码规范"排序
- 遗漏了论文论证的关键实验
- 部分方案会改变模型语义

**新计划理念:**
- 优先级按"论文质量"排序
- Blind corruption 是最高优先级
- 工程优化靠后,需要 benchmark

### 立即行动

**今天完成 (2小时):**
1. Blind corruption 实现
2. eval.seed 独立

**本周完成 (8小时):**
- 所有 P0 问题

**两周完成 (25小时):**
- P0 + P1 核心实验

---

**创建时间:** 2026-06-08  
**基于:** 你的专业评审意见  
**核心原则:** 论文质量优先,实验论证完整
