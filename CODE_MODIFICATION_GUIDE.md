# 代码修改详细指南

**基于:** 修正后的改进计划  
**目标:** 按优先级修复 P0 核心问题  
**预计时间:** 8 小时  
**创建日期:** 2026-06-08

---

## 📋 修改清单

### P0 优先级 (必须完成)
1. ✅ Blind Manifest Corruption - **已实现** (检查发现已有代码)
2. ⏳ eval.seed 独立
3. ⏳ Vocab Fingerprint 校验
4. ⏳ node_input_dim Strict 检查
5. ⏳ Resume Precheck 打印

---

## 🔴 修改 1: Blind Manifest Corruption

### 状态: ✅ 已实现

**检查结果:** 代码已经支持 blind 模式!

**当前代码位置:** `fusion/perturbations.py`

**已有功能:**
- 第 249 行: `_degrade_manifest()` 已支持 `blind` 参数
- 第 253-267 行: blind 模式逻辑已实现
- 第 312-316 行: `manifest_noisy_blind` 和 `manifest_shuffled_blind` 已支持

**验证方法:**
```python
# 测试 blind corruption
from fusion.perturbations import apply_aeg_view
import torch
from torch_geometric.data import Data

# 创建测试数据
data = Data(
    x=torch.randn(10, 128),
    pert_manifest=torch.tensor([0.0])
)

# 测试 blind variant
data_blind = apply_aeg_view(data, view="manifest_noisy_blind", strength=0.5)

print(f"Original pert_manifest: {data.pert_manifest.item()}")
print(f"Blind pert_manifest: {data_blind.pert_manifest.item()}")
# 预期: 两者相同 (blind 不修改 pert_manifest)
```

**需要做的:** 添加到训练配置中

**修改文件:** `config/experiments/aeg_robust/base.yaml`

**定位:** 第 65-76 行

**当前代码:**
```yaml
robust:
  train_aug: true
  manifest_donor_mode: cyclic
  perturb_prob: 0.5
  perturb_strengths: [0.1, 0.3, 0.5]
  train_views:
    - api_degraded
    - graph_degraded
    - api_graph_degraded
    - manifest_degraded
    - manifest_noisy
    - manifest_zeroed
    - manifest_shuffled
    - all_degraded
    - api_missing
    - graph_missing
    - manifest_missing
```

**修改为:**
```yaml
robust:
  train_aug: true
  manifest_donor_mode: cyclic
  perturb_prob: 0.5
  perturb_strengths: [0.1, 0.3, 0.5]
  train_views:
    - api_degraded
    - graph_degraded
    - api_graph_degraded
    - manifest_degraded
    - manifest_noisy
    - manifest_noisy_blind          # ✅ 新增
    - manifest_zeroed
    - manifest_shuffled
    - manifest_shuffled_blind       # ✅ 新增
    - all_degraded
    - api_missing
    - graph_missing
    - manifest_missing
```

**修改理由:**
1. **论文论证完整性:** 创新点3声称模型能"自主检测冲突"，必须在 blind 场景下验证
2. **审稿人会质疑:** 如果模型总是被告知 manifest 损坏，如何证明是"自主检测"？
3. **实验对比:** blind vs non-blind 的性能差异可以量化"conflict detection"的贡献

**同时修改评估配置:**

**定位:** 第 106-119 行

**当前代码:**
```yaml
eval:
  seed: 2026
  robust_eval: true
  perturb_strengths: [0.3, 0.5, 0.7]
  robust_views:
    - api_degraded
    - graph_degraded
    - api_graph_degraded
    - manifest_degraded
    - manifest_noisy
    - manifest_noisy_blind
    - manifest_zeroed
    - manifest_shuffled
    - manifest_shuffled_blind
    - all_degraded
    - api_missing
    - graph_missing
    - manifest_missing
```

**修改为:** (已经包含了 blind views)

**验证:** 确保第 111 和 115 行有 blind variants

---

## 🔴 修改 2: eval.seed 独立

### 问题分析

**当前问题:** `eval_seed` 从 `train_seed` 派生

**影响:** 多 seed 实验 (42, 52, 62) 时，评估扰动高度相关，影响实验公平性

### 修改文件 1: `fusion/train.py`

**定位:** 第 145-156 行 `_make_dataset` 函数

**当前代码:**
```python
def _make_dataset(cfg: dict[str, Any], split: str, *, aug: bool = False, view: str | None = None, strength: float | None = None) -> AEGDataset:
    data_cfg = cfg.get("data", {}) or {}
    split_cfg = data_cfg.get(split, {}) or {}
    pt_dir = split_cfg.get("pt_dir") or data_cfg.get(f"{split}_pt_dir")
    csv_path = split_cfg.get("csv") or split_cfg.get("label_csv") or data_cfg.get(f"{split}_csv")
    if not pt_dir:
        raise ValueError(f"data.{split}.pt_dir is required")
    if not csv_path:
        raise ValueError(f"data.{split}.csv is required")
    robust_cfg = cfg.get("robust", {}) or {}
    train_seed = int((cfg.get("train", {}) or {}).get("seed", 42))
    eval_seed = int((cfg.get("eval", {}) or {}).get("seed", train_seed + 100_000))  # ❌ 派生
    dataset_seed = eval_seed if view else train_seed
```

**修改为:**
```python
def _make_dataset(cfg: dict[str, Any], split: str, *, aug: bool = False, view: str | None = None, strength: float | None = None) -> AEGDataset:
    data_cfg = cfg.get("data", {}) or {}
    split_cfg = data_cfg.get(split, {}) or {}
    pt_dir = split_cfg.get("pt_dir") or data_cfg.get(f"{split}_pt_dir")
    csv_path = split_cfg.get("csv") or split_cfg.get("label_csv") or data_cfg.get(f"{split}_csv")
    if not pt_dir:
        raise ValueError(f"data.{split}.pt_dir is required")
    if not csv_path:
        raise ValueError(f"data.{split}.csv is required")
    robust_cfg = cfg.get("robust", {}) or {}
    train_seed = int((cfg.get("train", {}) or {}).get("seed", 42))
    # ✅ eval_seed 完全独立，不派生
    eval_seed = int((cfg.get("eval", {}) or {}).get("seed", 2026))
    # ✅ 评估扰动使用独立的 eval_seed
    dataset_seed = eval_seed if view else train_seed
```

**修改理由:**
1. **实验公平性:** 多 seed 实验应该使用独立的评估随机性
2. **可重现性:** eval_seed 固定为 2026，所有训练 seed 使用相同的评估扰动
3. **符合惯例:** 训练随机性和评估随机性应该独立控制

---

**定位 2:** 第 177-199 行 `_loader` 函数

**当前代码:**
```python
def _loader(cfg: dict[str, Any], dataset: AEGDataset, *, train: bool) -> DataLoader:
    train_cfg = cfg.get("train", {}) or {}
    batch_size = int(train_cfg.get("batch_size" if train else "eval_batch_size", train_cfg.get("batch_size", 24)))
    workers = int(train_cfg.get("num_workers", 0))
    generator = torch.Generator()
    train_seed = int(train_cfg.get("seed", 42))
    eval_seed = int((cfg.get("eval", {}) or {}).get("seed", train_seed + 100_000))  # ❌ 派生
    generator.manual_seed(train_seed if train else eval_seed)
```

**修改为:**
```python
def _loader(cfg: dict[str, Any], dataset: AEGDataset, *, train: bool) -> DataLoader:
    train_cfg = cfg.get("train", {}) or {}
    batch_size = int(train_cfg.get("batch_size" if train else "eval_batch_size", train_cfg.get("batch_size", 24)))
    workers = int(train_cfg.get("num_workers", 0))
    generator = torch.Generator()
    train_seed = int(train_cfg.get("seed", 42))
    # ✅ eval_seed 完全独立
    eval_seed = int((cfg.get("eval", {}) or {}).get("seed", 2026))
    generator.manual_seed(train_seed if train else eval_seed)
```

---

### 修改文件 2: `config/experiments/aeg_robust/base.yaml`

**定位:** 第 79-80 行

**当前代码:**
```yaml
eval:
  seed: 2026
  robust_eval: true
```

**确认:** 已经有独立的 eval.seed = 2026，无需修改

**但要确保所有继承 base.yaml 的配置都使用独立 seed**

---

**验证方法:**
```python
# 测试 seed 独立性
from fusion.train import load_config, _make_dataset

cfg = load_config("config/experiments/aeg_robust/base.yaml")

# 修改 train_seed
cfg["train"]["seed"] = 42
ds1 = _make_dataset(cfg, "val", aug=True, view="manifest_noisy", strength=0.5)

cfg["train"]["seed"] = 52  # 改变 train_seed
ds2 = _make_dataset(cfg, "val", aug=True, view="manifest_noisy", strength=0.5)

# 两个 dataset 的 seed 应该相同 (都是 eval_seed=2026)
print(f"Dataset 1 seed: {ds1.seed}")
print(f"Dataset 2 seed: {ds2.seed}")
# 预期: 两者相同
```

---

## 🔴 修改 3: Vocab Fingerprint 校验

### 新增文件: 在 `fusion/manifest_features.py` 中添加函数

**定位:** 文件末尾 (在已有函数之后)

**添加代码:**
```python
import csv
import hashlib
from pathlib import Path


def validate_vocab_for_train_csv(
    vocab: dict,
    train_csv_path: Path,
    *,
    strict: bool = True
) -> None:
    """验证 Manifest vocab 是否与当前 train CSV 匹配.
    
    Args:
        vocab: 加载的 vocab 字典
        train_csv_path: 当前训练集 CSV 路径
        strict: True 时不匹配会报错，False 时只警告
    
    Raises:
        ValueError: strict=True 且 vocab 与 train CSV 不匹配时
    """
    if not train_csv_path.exists():
        # 没有 train CSV，跳过检查
        return
    
    metadata = vocab.get("metadata") or {}
    
    # 读取当前 train CSV 的 sample IDs
    current_ids = set()
    try:
        with train_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 尝试多个可能的 ID 列名
                sid = ""
                for id_field in ["sha256", "apk_sha256", "id", "sid", "sample_id"]:
                    if id_field in row:
                        sid = str(row[id_field] or "").strip().lower()
                        if sid:
                            break
                if sid:
                    current_ids.add(sid)
    except Exception as e:
        if strict:
            raise ValueError(f"Failed to read train CSV {train_csv_path}: {e}")
        return
    
    if not current_ids:
        # CSV 为空或无法读取，跳过检查
        return
    
    current_count = len(current_ids)
    current_fingerprint = hashlib.sha256(
        ";".join(sorted(current_ids)).encode("utf-8")
    ).hexdigest()
    
    # 对比 metadata
    expected_count = int(metadata.get("source_id_count", -1))
    expected_fp = str(metadata.get("source_id_fingerprint", ""))
    
    if expected_count <= 0 or not expected_fp:
        # Vocab metadata 缺失，可能是旧版本
        if strict:
            raise ValueError(
                f"Vocab metadata is incomplete. "
                f"Rebuild vocab with current scripts."
            )
        return
    
    # 检查是否匹配
    count_match = (expected_count == current_count)
    fp_match = (expected_fp == current_fingerprint)
    
    if not count_match or not fp_match:
        msg = (
            f"Manifest vocab may not match current train CSV:\n"
            f"  Vocab was built from {expected_count} samples (fingerprint: {expected_fp[:16]}...)\n"
            f"  Current train CSV has {current_count} samples (fingerprint: {current_fingerprint[:16]}...)\n"
            f"  Rebuild vocab with: python scripts/build_aeg_pts_direct.py --rebuild-vocab"
        )
        if strict:
            raise ValueError(msg)
        else:
            import warnings
            warnings.warn(msg)
```

**修改理由:**
1. **数据完整性:** 避免用错 vocab 导致特征向量错位
2. **Silent fail 预防:** 如果 vocab 不匹配，应该明确报错而不是静默失败
3. **可追溯性:** fingerprint 可以追溯 vocab 是从哪个 train split 生成的

---

### 修改文件: `fusion/train.py`

**定位:** `run()` 函数开头 (约第 480-487 行)

**当前代码:**
```python
def run(cfg: dict[str, Any]) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO)
    train_cfg = cfg.get("train", {}) or {}
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    device = _device(cfg)
    out_dir = Path(train_cfg.get("output_dir", "results/aeg_robust/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
```

**修改为:**
```python
def run(cfg: dict[str, Any]) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO)
    train_cfg = cfg.get("train", {}) or {}
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    device = _device(cfg)
    out_dir = Path(train_cfg.get("output_dir", "results/aeg_robust/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ✅ 验证 Manifest vocab 与 train CSV 是否匹配
    data_cfg = cfg.get("data", {}) or {}
    train_label_csv = (data_cfg.get("train", {}) or {}).get("csv")
    if train_label_csv:
        train_csv_path = Path(train_label_csv)
        # 尝试加载 vocab (如果存在)
        manifest_cfg = cfg.get("manifest", {}) or {}
        vocab_path_str = manifest_cfg.get("vocab_path", "config/manifest_vocab_aeg.yaml")
        vocab_path = Path(vocab_path_str)
        if vocab_path.exists():
            from fusion.manifest_features import validate_vocab_for_train_csv
            try:
                vocab = load_yaml(vocab_path)
                validate_vocab_for_train_csv(
                    vocab,
                    train_csv_path,
                    strict=bool(data_cfg.get("strict_vocab_check", True))
                )
                LOGGER.info("✓ Manifest vocab matches current train CSV")
            except ValueError as e:
                LOGGER.error(f"Vocab validation failed: {e}")
                raise
```

**定位:** 需要添加 import (文件开头)

**添加:**
```python
# 在文件开头的 import 部分添加
import warnings  # 如果还没有
```

---

**验证方法:**
```bash
# 测试 vocab 校验
python -m fusion.train --config config/experiments/aeg_robust/base.yaml

# 应该看到日志:
# ✓ Manifest vocab matches current train CSV
```

---

## 🔴 修改 4: node_input_dim Strict 检查

### 修改文件: `fusion/model.py`

**定位:** 第 276-283 行 `_initial_node_state` 方法

**当前代码:**
```python
def _initial_node_state(self, data: Batch) -> torch.Tensor:
    x = data.x.float()
    if x.size(1) != self.input_proj.in_features:
        if x.size(1) < self.input_proj.in_features:
            pad = x.new_zeros((x.size(0), self.input_proj.in_features - x.size(1)))
            x = torch.cat([x, pad], dim=-1)
        else:
            x = x[:, : self.input_proj.in_features]
    node_type = data.node_type.long().clamp(0, NUM_NODE_TYPES - 1)
```

**修改为:**
```python
def _initial_node_state(self, data: Batch) -> torch.Tensor:
    x = data.x.float()
    expected_dim = self.input_proj.in_features
    actual_dim = x.size(1)
    
    if actual_dim != expected_dim:
        # ✅ 默认严格检查，维度不匹配时报错
        # 只有明确配置 auto_adjust_input_dim=true 才容忍
        if not getattr(self, "auto_adjust_input_dim", False):
            raise RuntimeError(
                f"Node feature dimension mismatch: "
                f"model expects {expected_dim}, but PT file has {actual_dim}. "
                f"This usually means the PT files were generated with a different "
                f"aeg.node_feature_dim setting. "
                f"Check that extraction config matches training config, "
                f"or set model.auto_adjust_input_dim=true to allow padding/truncation."
            )
        
        # Auto adjust (仅在明确配置时)
        if actual_dim < expected_dim:
            pad = x.new_zeros((x.size(0), expected_dim - actual_dim))
            x = torch.cat([x, pad], dim=-1)
        else:
            x = x[:, :expected_dim]
    
    node_type = data.node_type.long().clamp(0, NUM_NODE_TYPES - 1)
```

**修改理由:**
1. **与 strict contract 理念一致:** 维度不匹配是配置错误，应该报错
2. **避免 silent fail:** 静默 pad/truncate 可能掩盖 PT 版本不匹配
3. **提供清晰错误信息:** 告诉用户如何修复
4. **保留灵活性:** 通过配置可以开启 auto adjust

---

### 修改文件: `config/experiments/aeg_robust/base.yaml`

**定位:** 第 30-46 行 model 配置

**当前代码:**
```yaml
model:
  node_input_dim: 128
  hidden_dim: 128
  layers: 2
  dropout: 0.15
  num_latents: 16
  fusion_mode: latent
  use_relation_types: true
  use_node_types: true
  use_node_source: true
  use_edge_source: true
  use_node_quality: true
  use_edge_quality: true
  source_bias_weight: 1.0
  reliability_bias_weight: 1.0
  conflict_bias_weight: 0.5
  num_classes: 2
```

**添加:**
```yaml
model:
  node_input_dim: 128
  hidden_dim: 128
  layers: 2
  dropout: 0.15
  num_latents: 16
  fusion_mode: latent
  use_relation_types: true
  use_node_types: true
  use_node_source: true
  use_edge_source: true
  use_node_quality: true
  use_edge_quality: true
  source_bias_weight: 1.0
  reliability_bias_weight: 1.0
  conflict_bias_weight: 0.5
  num_classes: 2
  auto_adjust_input_dim: false  # ✅ 新增: 默认 false，严格检查
```

**修改理由:** 明确配置严格检查模式

---

**验证方法:**
```python
# 测试维度不匹配时是否报错
from fusion.model import build_model
from fusion.train import load_config
import torch
from torch_geometric.data import Batch

cfg = load_config("config/experiments/aeg_robust/base.yaml")
model = build_model(cfg, node_input_dim=128)

# 创建维度不匹配的测试数据
data = Batch(
    x=torch.randn(10, 256),  # ❌ 256 != 128
    node_type=torch.zeros(10, dtype=torch.long),
    node_source=torch.zeros(10, dtype=torch.long),
    node_quality=torch.ones(10, 1),
    node_semantic=torch.zeros(10, 12),
)

try:
    logits, extra = model(data)
    print("❌ 应该报错但没有!")
except RuntimeError as e:
    print(f"✅ 正确报错: {e}")
```

---

## 🔴 修改 5: Resume Precheck 打印

### 修改文件: `scripts/build_aeg_pts_direct.py`

**定位:** 主流程中，在开始处理 jobs 之前

**需要找到主流程位置 (通常在 main() 函数末尾)**

让我先读取文件结构:

**当前代码:** (需要在 jobs 循环之前添加)

**添加位置:** 在开始生成 PT 之前

**添加代码:**
```python
# ✅ Resume precheck: 统计可 resume 的数量
if cfg["resume"]:
    print("\n" + "=" * 80)
    print("📊 Resume Precheck")
    print("=" * 80)
    
    resumable_count = 0
    pending_count = 0
    
    # 快速扫描统计
    for job in tqdm(jobs, desc="Checking resumable PT files", unit="sample"):
        resume_row = _resume_existing(job, cfg)
        if resume_row is not None:
            resumable_count += 1
        else:
            pending_count += 1
    
    total_count = len(jobs)
    resumable_pct = 100 * resumable_count / total_count if total_count > 0 else 0
    
    print(f"\n✅ Resumable PT files: {resumable_count}/{total_count} ({resumable_pct:.1f}%)")
    print(f"⏳ Pending generation: {pending_count}/{total_count}")
    
    if resumable_count == total_count:
        print("\n🎉 All PT files can be resumed!")
        print("   Generation will complete in seconds.")
    elif resumable_count > 0:
        # 估算节省的时间 (假设每个 PT 平均生成需要 30 秒)
        saved_hours = resumable_count * 30 / 3600
        print(f"\n⚡ Will reuse {resumable_count} existing PT files")
        print(f"   Estimated time saved: ~{saved_hours:.1f} hours")
    else:
        print("\n⚠️  No existing PT files found, will generate all from scratch")
    
    print("=" * 80 + "\n")
```

**修改理由:**
1. **用户体验:** 让用户知道有多少 PT 可以 resume
2. **时间预估:** 告诉用户能节省多少时间
3. **透明度:** 避免用户不知道脚本在做什么

**注意:** 需要导入 tqdm (应该已经有)

---

## 📝 修改总结

### 需要修改的文件清单

| 文件 | 修改内容 | 行数 | 难度 |
|------|---------|------|------|
| `config/experiments/aeg_robust/base.yaml` | 添加 blind views | 2 行 | ⭐ |
| `fusion/train.py` | eval.seed 独立 (2处) | 4 行 | ⭐ |
| `fusion/manifest_features.py` | 添加 vocab 校验函数 | 70 行 | ⭐⭐ |
| `fusion/train.py` | 调用 vocab 校验 | 15 行 | ⭐⭐ |
| `fusion/model.py` | node_input_dim strict | 15 行 | ⭐⭐ |
| `config/experiments/aeg_robust/base.yaml` | 添加 auto_adjust 配置 | 1 行 | ⭐ |
| `scripts/build_aeg_pts_direct.py` | Resume precheck | 30 行 | ⭐⭐ |

**总行数:** 约 137 行  
**预计时间:** 3-4 小时

---

## ✅ 验证计划

### 1. Blind Corruption 验证
```bash
# 训练包含 blind views
python -m fusion.train --config config/experiments/aeg_robust/base.yaml

# 检查日志中是否有 manifest_noisy_blind 和 manifest_shuffled_blind
```

### 2. eval.seed 验证
```bash
# 多 seed 实验
python run.py full_seeds

# 检查三个 seed 的 val metrics 是否合理
```

### 3. Vocab 校验验证
```bash
# 正常训练应该通过
python -m fusion.train --config config/experiments/aeg_robust/base.yaml

# 应该看到: ✓ Manifest vocab matches current train CSV
```

### 4. node_input_dim 验证
```bash
# 创建测试脚本测试维度不匹配
python scripts/test_dim_mismatch.py
```

### 5. Resume Precheck 验证
```bash
# 运行 PT 生成
python scripts/build_aeg_pts_direct.py \
  --config config/extract_aeg.yaml \
  --no-rebuild-vocab \
  --resume \
  --workers 8

# 应该看到 precheck 统计
```

---

## 🎯 修改优先级

### 必须立即做 (今天)
1. ✅ Blind views 配置 (已完成检查,只需添加到 yaml)
2. ⏳ eval.seed 独立

### 明天完成
3. ⏳ Vocab fingerprint 校验
4. ⏳ node_input_dim strict

### 本周完成
5. ⏳ Resume precheck

---

## 📞 遇到问题?

### 常见问题

**Q1: 找不到某个函数的位置?**
A: 使用 grep 搜索:
```bash
grep -n "def _make_dataset" fusion/train.py
```

**Q2: 不确定代码是否正确?**
A: 先备份:
```bash
cp fusion/train.py fusion/train.py.backup
```

**Q3: 修改后训练报错?**
A: 检查语法错误:
```bash
python -m py_compile fusion/train.py
```

---

**创建时间:** 2026-06-08  
**预计完成:** 2026-06-09  
**状态:** 待执行

祝修改顺利! 🚀
