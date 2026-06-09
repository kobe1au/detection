# PT文件Resume问题诊断与解决方案

## 问题描述

**症状:** 执行 `python scripts/build_aeg_pts_direct.py --config config/extract_aeg.yaml --no-rebuild-vocab --resume --workers 8` 时,脚本无法识别已生成的 PT 文件,从头开始重新生成。

**实际情况:**
- 已有 10,334 个 train PT 文件在 `D:/pts_aeg/train/`
- 使用 `--resume` 参数但仍然重新生成
- 扫描索引文件存在: `D:/pts_aeg/aeg_apk_scan_index.csv`

---

## 🔍 根本原因分析

根据代码审查 (`build_aeg_pts_direct.py:827-848`),resume 机制依赖于 **build fingerprint 匹配**:

```python
def _resume_existing(job, cfg):
    if not cfg["resume"]:
        return None
    out_path = _out_path(job, cfg)
    if not out_path.exists():
        return None
    try:
        existing = torch.load(out_path, map_location="cpu")
        expected_fingerprint = str(cfg.get("build_fingerprint") or "")
        if not expected_fingerprint:
            return None
        validate_aeg_payload(
            existing,
            expected_build_fingerprint=expected_fingerprint,  # ⚠️ 关键检查
            expected_node_feature_dim=cfg["node_feature_dim"],
        )
        return _index_row(job, out_path, "ok", "resume")
    except Exception:
        return None  # ⚠️ 任何异常都会导致重新生成
```

**Build fingerprint 包含:**
1. **源代码哈希** (8个Python文件)
2. **配置参数** (32个超参数)
3. **Manifest词汇表**
4. **Schema版本和Contract版本**

**结论:** 如果你修改了代码(提交 7fc8e39 "代码大改9"),build fingerprint 会改变,导致所有旧 PT 文件被认为"过时"而重新生成。

---

## ✅ 解决方案

### 方案 1: 继续重新生成 (推荐)

**适用场景:** 代码修改可能影响 PT 文件内容

**操作:**
```bash
# 直接让脚本完整运行
python scripts/build_aeg_pts_direct.py \
  --config config/extract_aeg.yaml \
  --no-rebuild-vocab \
  --resume \
  --workers 16
```

**优化建议:**
- 增加 workers 数量 (如果 CPU 允许):
  ```yaml
  # config/extract_aeg.yaml
  execution:
    workers: 16  # 根据CPU核心数调整
  ```

---

### 方案 2: 检查是否真的需要重新生成

**检查哪些文件被修改:**
```bash
# 查看最近几次提交修改了哪些文件
git diff 806c98f..7fc8e39 --name-only

# 检查是否修改了PT生成相关代码
git diff 806c98f..7fc8e39 -- \
  scripts/build_aeg_pts_direct.py \
  fusion/constants.py \
  fusion/aeg_builder.py \
  fusion/manifest_features.py \
  extract/extract_graph_api.py
```

**如果只修改了训练代码** (`model.py`, `losses.py`, `train.py`, `perturbations.py`),可以跳过重新生成,直接训练。

---

### 方案 3: 验证旧PT文件是否兼容

```bash
# 测试加载旧PT文件
python -c "
import torch
from pathlib import Path
from fusion.dataset import AEGDataset
from fusion.train import load_config

# 加载一个旧PT文件
pt_path = Path('D:/pts_aeg/train').glob('*.pt').__next__()
print(f'Testing: {pt_path.name}')

payload = torch.load(pt_path, map_location='cpu')
print(f'✓ Build fingerprint: {payload.get(\"aeg_build_fingerprint\", \"MISSING\")[:16]}...')
print(f'✓ Schema version: {payload.get(\"schema_version\")}')
print(f'✓ Node feature dim: {payload[\"node_x\"].shape[1]}')
print(f'✓ Edge types: {payload[\"edge_type\"].unique().tolist()}')

# 尝试加载为Dataset
ds = AEGDataset(
    'D:/pts_aeg/train',
    'results/labels/train.csv',
    split='train',
    train_aug=False,
    validate_payload_on_load=False,  # 跳过验证测试能否加载
)
print(f'✓ Dataset loaded: {len(ds)} samples')

# 获取一个batch
item = ds[0]
print(f'✓ Sample loaded: x.shape={item[\"clean\"].x.shape}')
print()
print('✅ 旧PT文件可以正常加载!')
print('⚠️ 但可能与新代码不完全兼容,建议重新生成')
"
```

---

## 🔧 调试命令

### 检查PT文件状态
```bash
# 统计已生成的PT文件数量
ls D:/pts_aeg/train/*.pt | wc -l

# 检查最新生成的PT文件
ls -lt D:/pts_aeg/train/*.pt | head -5

# 检查扫描索引
head -5 D:/pts_aeg/aeg_apk_scan_index.csv
grep -c "^train," D:/pts_aeg/aeg_apk_scan_index.csv
```

### 监控生成进度
```bash
# 实时监控PT文件数量变化
watch -n 5 'ls D:/pts_aeg/train/*.pt | wc -l'
```

---

## ❓ 常见问题

### Q1: 为什么修改训练代码也要重新生成PT?
**A:** 如果只修改了 `model.py`, `losses.py`, `train.py`,理论上**不需要**重新生成 PT。但如果修改了 `constants.py` (例如增加节点类型、边类型),则必须重新生成。

### Q2: 生成速度太慢怎么办?
**A:** 
1. 增加 `workers` 到 CPU 核心数
2. 使用 SSD 存储 PT 文件
3. 减少 `max_methods_per_apk` 和 `max_events_per_apk`

### Q3: 如何只生成train而不生成val/test?
**A:**
```yaml
# config/extract_aeg_train_only.yaml
data:
  splits: [train]  # 只生成train
  split_dirs:
    train: E:/train
  out_dirs:
    train: D:/pts_aeg/train
  label_csvs:
    train: results/labels/train.csv
```

---

## 📝 总结

**你的问题根本原因:**
- 代码修改 ("代码大改9") 导致 build fingerprint 改变
- Resume 机制检测到 fingerprint 不匹配,拒绝使用旧 PT 文件
- 脚本从头开始重新生成

**最佳解决方案:**
- ✅ **让脚本完整运行**,重新生成所有 PT 文件
- ✅ 优化 `workers` 参数加速生成(建议16)
- ✅ 确保数据一致性和可重现性

**快速测试方案:**
- 先检查是否修改了PT生成相关代码
- 如果只改了训练代码,直接训练测试
- 如果训练出错,再重新生成PT

---

**创建时间:** 2026-06-08  
**适用版本:** commit 7fc8e39 "代码大改9"
