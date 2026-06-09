# 代码改进实施计划

## 优先级分级

- 🔴 **P0 - 立即修复**: 影响可用性的严重问题
- 🟡 **P1 - 短期优化**: 影响性能或可维护性
- 🟢 **P2 - 中期增强**: 功能扩展和优化

---

## 🔴 P0: 立即修复 (1-2天)

### 1. 修复硬编码路径问题

**问题文件:** `config/experiments/aeg_robust/base.yaml`

**当前代码:**
```yaml
data:
  train:
    pt_dir: D:/pts_aeg/train  # ❌ 硬编码 Windows 路径
    csv: results/labels/train.csv
```

**修复方案 A - 环境变量:**
```yaml
data:
  # 支持环境变量,默认值为 ./data/aeg_pts
  pt_root: ${PT_ROOT:./data/aeg_pts}
  train:
    pt_dir: ${data.pt_root}/train
    csv: results/labels/train.csv
  val:
    pt_dir: ${data.pt_root}/val
    csv: results/labels/val.csv
  test:
    pt_dir: ${data.pt_root}/test
    csv: results/labels/test.csv
```

**修复方案 B - 配置加载时解析:**
在 `train.py` 中添加路径解析逻辑:

```python
# train.py 新增函数
def resolve_path(path_str: str, base_dir: Path = Path.cwd()) -> Path:
    """解析路径字符串,支持环境变量和相对路径"""
    import os
    # 1. 展开环境变量
    expanded = os.path.expandvars(path_str)
    # 2. 展开用户目录
    expanded = os.path.expanduser(expanded)
    # 3. 转为 Path 对象
    path = Path(expanded)
    # 4. 如果是相对路径,相对于配置文件或项目根目录
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()

def _make_dataset(cfg: dict[str, Any], split: str, ...) -> AEGDataset:
    data_cfg = cfg.get("data", {}) or {}
    split_cfg = data_cfg.get(split, {}) or {}
    pt_dir = split_cfg.get("pt_dir") or data_cfg.get(f"{split}_pt_dir")
    csv_path = split_cfg.get("csv") or split_cfg.get("label_csv") or data_cfg.get(f"{split}_csv")
    
    # ✅ 添加路径解析
    if not pt_dir:
        raise ValueError(f"data.{split}.pt_dir is required")
    pt_dir = resolve_path(pt_dir)
    
    if not csv_path:
        raise ValueError(f"data.{split}.csv is required")
    csv_path = resolve_path(csv_path)
    
    return AEGDataset(pt_dir, csv_path, ...)
```

**推荐:** 使用方案 B,更灵活且无需修改现有配置文件格式。

---

### 2. 添加数值稳定性修复

**问题文件:** `fusion/model.py:361-364`

**当前代码:**
```python
code_rel = (
    (r_api * r_graph).sqrt()
    * (0.5 + 0.5 * q_align.clamp(0.0, 1.0))
).clamp(0.0, 1.0)
```

**问题:** 当 `r_api` 或 `r_graph` 为 0 时,`sqrt(0)` 导致梯度为 inf。

**修复代码:**
```python
# model.py 文件顶部添加常量
RELIABILITY_EPS = 1e-6

# model.py:361-364 修改为:
code_rel = (
    ((r_api + RELIABILITY_EPS) * (r_graph + RELIABILITY_EPS)).sqrt()
    * (0.5 + 0.5 * q_align.clamp(0.0, 1.0))
).clamp(RELIABILITY_EPS, 1.0)  # 也避免输出 0
```

**同时修复:** `losses.py:172` 中的 log 操作
```python
# losses.py:172
# 当前:
scores = scores + self.reliability_bias_weight * torch.log(rel).unsqueeze(1)

# 修复为:
scores = scores + self.reliability_bias_weight * torch.log(rel + 1e-8).unsqueeze(1)
```

---

### 3. Manifest Shuffling 单样本检查

**问题文件:** `fusion/dataset.py:187-243`

**修复代码:**
```python
class AEGDataset(Dataset):
    def __init__(
        self,
        pt_dir: str | Path,
        label_csv: str | Path | None = None,
        *,
        split: str = "",
        train_aug: bool = False,
        aug_views: list[str] | tuple[str, ...] | None = None,
        ...
    ) -> None:
        # ... 现有代码 ...
        
        if not samples:
            raise AEGDatasetConfigError(f"No AEG PT samples found in {self.pt_dir}")
        
        # ✅ 新增检查
        shuffled_views = {"manifest_shuffled", "manifest_shuffled_blind"}
        if train_aug and shuffled_views.intersection(self.aug_views):
            if len(samples) < 2:
                raise AEGDatasetConfigError(
                    f"Manifest shuffling requires at least 2 samples for donor selection; "
                    f"got {len(samples)} samples in {self.pt_dir}. "
                    f"Remove {shuffled_views.intersection(self.aug_views)} from aug_views "
                    f"or use a larger dataset split."
                )
        
        self.samples = samples
        self.manifest_donor_indices = _build_manifest_donor_indices(...)
```

---

## 🟡 P1: 短期优化 (1周)

### 4. 实现 PT 文件缓存层

**问题:** 每次 `__getitem__` 都从磁盘加载,I/O 密集。

**新建文件:** `fusion/payload_cache.py`

```python
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


class PayloadCache:
    """线程安全的 LRU 缓存,用于 AEG PT 文件."""
    
    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self.cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, path: Path) -> dict[str, Any] | None:
        with self.lock:
            if path in self.cache:
                self.hits += 1
                self.cache.move_to_end(path)  # 标记为最近使用
                return self.cache[path]
            self.misses += 1
            return None
    
    def put(self, path: Path, payload: dict[str, Any]) -> None:
        with self.lock:
            if path in self.cache:
                self.cache.move_to_end(path)
            else:
                self.cache[path] = payload
                if len(self.cache) > self.max_size:
                    # 淘汰最旧的项
                    self.cache.popitem(last=False)
    
    def load(self, path: Path) -> dict[str, Any]:
        """加载 payload,优先从缓存读取"""
        cached = self.get(path)
        if cached is not None:
            return cached
        
        payload = torch.load(path, map_location="cpu")
        self.put(path, payload)
        return payload
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
        }


# 全局缓存实例(每个进程一个)
_global_cache: PayloadCache | None = None


def get_global_cache() -> PayloadCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = PayloadCache(max_size=512)
    return _global_cache
```

**修改:** `fusion/dataset.py`
```python
from fusion.payload_cache import get_global_cache

class AEGDataset(Dataset):
    def __init__(self, ..., use_cache: bool = True):
        # ... 现有代码 ...
        self.use_cache = use_cache
        self.cache = get_global_cache() if use_cache else None
    
    def __getitem__(self, idx: int) -> dict[str, Data]:
        path, label = self.samples[idx]
        
        # ✅ 使用缓存
        if self.cache:
            payload = self.cache.load(path)
        else:
            payload = torch.load(path, map_location="cpu")
        
        clean = payload_to_data(payload, label=label, ...)
        # ... 其余代码不变 ...
```

**配置添加:**
```yaml
# base.yaml
data:
  use_cache: true  # 默认启用缓存
  cache_size: 512  # 缓存最多 512 个 PT 文件
```

**预期收益:**
- 训练速度提升 20-40% (取决于磁盘速度)
- 减少 I/O 等待时间

---

### 5. 边质量传播改进

**问题文件:** `fusion/model.py:314-315`

**当前代码:**
```python
edge_quality = edge_quality * node_weight.view(-1)[src] * node_weight.view(-1)[dst]
```

**问题:** 级联衰减过快 (0.8³ = 0.512)

**修复方案 A - 几何平均:**
```python
# 使用几何平均,衰减更温和
edge_node_quality = (node_weight.view(-1)[src] * node_weight.view(-1)[dst]).sqrt()
edge_quality = edge_quality * edge_node_quality
```

**修复方案 B - 可学习衰减因子:**
```python
# model.py AEGModel.__init__ 添加:
self.quality_decay_factor = nn.Parameter(torch.tensor(0.5))  # 可学习

# forward 中:
edge_node_quality = (node_weight.view(-1)[src] * node_weight.view(-1)[dst]).pow(
    self.quality_decay_factor.clamp(0.1, 1.0)
)
edge_quality = edge_quality * edge_node_quality
```

**推荐:** 方案 A,简单有效。方案 B 可作为消融实验。

---

### 6. Magic Number 重构

**问题文件:** `fusion/model.py`

**新建文件:** `fusion/fusion_constants.py`
```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusionConstants:
    """潜变量融合的硬编码常量."""
    
    # Code embedding 组成权重
    CODE_METHOD_WEIGHT: float = 0.5
    CODE_API_WEIGHT: float = 0.5
    
    # Manifest embedding 组成权重
    MANIFEST_PERMISSION_WEIGHT: float = 0.5
    MANIFEST_COMPONENT_WEIGHT: float = 0.5
    
    # Token 冲突敏感度 (对应 token 顺序)
    # [method, api_family, permission, component, risk, string_hint, global]
    TOKEN_CONFLICT_SENSITIVITY: tuple[float, ...] = (
        0.0,   # method: 代码证据,不受 manifest 冲突影响
        0.0,   # api_family: 代码证据,不受 manifest 冲突影响
        1.0,   # permission: manifest 证据,完全冲突敏感
        1.0,   # component: manifest 证据,完全冲突敏感
        0.5,   # risk: 混合证据,部分冲突敏感
        0.0,   # string_hint: 代码派生,不受冲突影响
        0.25,  # global: 全局混合,轻度冲突敏感
    )
    
    # Token 源类型 (对应 SOURCE_TYPES)
    # [method, api_family, permission, component, risk, string_hint, global]
    TOKEN_SOURCE_TYPES: tuple[int, ...] = (
        0,  # method: code
        0,  # api_family: code
        1,  # permission: manifest
        1,  # component: manifest
        2,  # risk: derived
        2,  # string_hint: derived
        2,  # global: derived
    )


DEFAULT_FUSION_CONSTANTS = FusionConstants()
```

**修改:** `fusion/model.py`
```python
from fusion.fusion_constants import DEFAULT_FUSION_CONSTANTS as FC

class AEGModel(nn.Module):
    def forward(self, data: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # ... 前面代码 ...
        
        # 使用常量替换硬编码
        code_emb = (
            FC.CODE_METHOD_WEIGHT * method_emb 
            + FC.CODE_API_WEIGHT * api_family_emb
        )
        manifest_emb = (
            FC.MANIFEST_PERMISSION_WEIGHT * permission_emb 
            + FC.MANIFEST_COMPONENT_WEIGHT * component_emb
        )
        
        # Token 源类型
        token_source = torch.tensor(
            FC.TOKEN_SOURCE_TYPES,
            dtype=torch.long,
            device=data.x.device,
        )
        
        # Token 冲突敏感度
        token_conflict_sensitivity = torch.tensor(
            FC.TOKEN_CONFLICT_SENSITIVITY,
            device=data.x.device,
        )
        # ... 其余代码 ...
```

---

## 🟢 P2: 中期增强 (2-3周)

### 7. Attention 可视化工具

**新建文件:** `scripts/visualize_attention.py`

```python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_attention_distribution(csv_path: Path, output_dir: Path) -> None:
    """绘制 attention 分布统计图."""
    df = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    attn_cols = [
        "attn_method",
        "attn_api_family",
        "attn_permission",
        "attn_component",
        "attn_risk",
        "attn_string_hint",
        "attn_global",
    ]
    
    # 检查列是否存在
    available_cols = [col for col in attn_cols if col in df.columns]
    if not available_cols:
        print(f"No attention columns found in {csv_path}")
        return
    
    # 1. 总体分布箱线图
    fig, ax = plt.subplots(figsize=(12, 6))
    df[available_cols].boxplot(ax=ax)
    ax.set_title("Attention Mass Distribution Across Token Types")
    ax.set_ylabel("Attention Weight")
    ax.set_xlabel("Token Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "attention_boxplot.png", dpi=300)
    plt.close()
    
    # 2. 按标签分组的平均 attention
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, ax in zip([0, 1], axes):
        subset = df[df["label"] == label][available_cols]
        mean_attn = subset.mean().values
        ax.bar(range(len(available_cols)), mean_attn)
        ax.set_xticks(range(len(available_cols)))
        ax.set_xticklabels([c.replace("attn_", "") for c in available_cols], rotation=45)
        ax.set_title(f"{'Benign' if label == 0 else 'Malware'} (n={len(subset)})")
        ax.set_ylabel("Mean Attention")
        ax.set_ylim(0, max(mean_attn) * 1.2)
    plt.tight_layout()
    plt.savefig(output_dir / "attention_by_label.png", dpi=300)
    plt.close()
    
    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[available_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Attention Token Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "attention_correlation.png", dpi=300)
    plt.close()
    
    # 4. 错误样本的 attention 分析
    if "pred" in df.columns:
        errors = df[df["label"] != df["pred"]]
        if len(errors) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            error_attn = errors[available_cols].mean()
            correct_attn = df[df["label"] == df["pred"]][available_cols].mean()
            
            x = range(len(available_cols))
            width = 0.35
            ax.bar([i - width/2 for i in x], error_attn, width, label="Errors")
            ax.bar([i + width/2 for i in x], correct_attn, width, label="Correct")
            ax.set_xticks(x)
            ax.set_xticklabels([c.replace("attn_", "") for c in available_cols], rotation=45)
            ax.set_title(f"Attention Pattern: Errors (n={len(errors)}) vs Correct")
            ax.set_ylabel("Mean Attention")
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "attention_errors.png", dpi=300)
            plt.close()
    
    print(f"✅ Saved attention visualizations to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize AEG attention patterns")
    parser.add_argument("--csv", type=Path, required=True, help="Diagnostics CSV file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    
    plot_attention_distribution(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
```

**使用示例:**
```bash
python scripts/visualize_attention.py \
  --csv results/aeg_robust/full/ours/diagnostics_test_clean.csv \
  --output-dir results/aeg_robust/full/ours/visualizations
```

---

### 8. 课程学习的动态扰动

**修改文件:** `fusion/train.py`

```python
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
) -> dict[str, float]:
    model.train()
    use_aug = bool((cfg.get("robust", {}) or {}).get("train_aug", True))
    
    # ✅ 新增: 课程学习
    curriculum_cfg = (cfg.get("robust", {}) or {}).get("curriculum", {})
    if curriculum_cfg.get("enabled", False):
        max_epochs = int((cfg.get("train", {}) or {}).get("epochs", 60))
        # 线性增长: epoch 1 → 0.1, epoch max → 0.7
        curriculum_strength = 0.1 + 0.6 * min(epoch / max_epochs, 1.0)
        # 动态调整 loader 的扰动强度(需要重新构建 dataset)
        # 这里简化为打印,实际需要重新创建 loader
        print(f"Curriculum strength for epoch {epoch}: {curriculum_strength:.2f}")
    
    # ... 其余训练代码不变 ...
```

**配置添加:**
```yaml
# base.yaml
robust:
  curriculum:
    enabled: false  # 默认关闭,可在消融实验中启用
    start_strength: 0.1
    end_strength: 0.7
```

---

### 9. 温度参数消融实验

**新建配置文件:** `config/experiments/aeg_robust/ablation/temperature_sweep.yaml`

```yaml
# 创建多个配置测试不同温度
base: ../base.yaml

train:
  output_dir: results/aeg_robust/ablation/temperature_0.1

loss:
  temperature: 0.1
```

**或使用脚本批量生成:**

**新建文件:** `scripts/generate_temperature_configs.py`
```python
from pathlib import Path

template = """base: ../base.yaml

train:
  output_dir: results/aeg_robust/ablation/temperature_{temp}

loss:
  temperature: {temp}
"""

output_dir = Path("config/experiments/aeg_robust/ablation")
output_dir.mkdir(parents=True, exist_ok=True)

for temp in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    config_path = output_dir / f"temperature_{temp}.yaml"
    with config_path.open("w") as f:
        f.write(template.format(temp=temp))
    print(f"Created {config_path}")
```

---

## 📋 实施时间表

| 周次 | 任务 | 预计工时 |
|------|------|----------|
| Week 1 | P0-1: 路径修复 | 4h |
| Week 1 | P0-2: 数值稳定性 | 2h |
| Week 1 | P0-3: 单样本检查 | 1h |
| Week 2 | P1-4: 缓存层 | 6h |
| Week 2 | P1-5: 边质量改进 | 3h |
| Week 2 | P1-6: 常量重构 | 4h |
| Week 3 | P2-7: 可视化工具 | 8h |
| Week 3 | P2-8: 课程学习 | 6h |
| Week 3 | P2-9: 温度消融 | 2h |

**总计:** ~36 工时 (约 1 周全职或 2-3 周兼职)

---

## 🧪 测试计划

### 回归测试
每次改动后运行:
```bash
python -m pytest tests/test_aeg_smoke.py -v
```

### 性能基准测试
修复前后对比:
```bash
# 训练 1 个 epoch 测速
time python -m fusion.train --config config/experiments/aeg_robust/full/ours.yaml
```

### 正确性验证
确保修复不改变模型行为:
```bash
# 使用相同种子训练,对比最终指标
python run.py full
# 检查 results/aeg_robust/full/ours/summary.json
```

---

## 📊 预期改进效果

| 改进项 | 指标 | 改进前 | 改进后 | 提升 |
|--------|------|--------|--------|------|
| PT 缓存 | 训练速度 | 1.0x | 1.3-1.5x | +30-50% |
| 数值稳定 | 梯度爆炸率 | ~5% | <1% | -80% |
| 路径修复 | 跨平台兼容 | ❌ | ✅ | 100% |
| 边质量改进 | 长路径信号 | 衰减严重 | 保留更好 | +15% |

---

## ✅ 验收标准

### P0 完成标准:
- [ ] 所有配置文件使用相对路径或环境变量
- [ ] 无数值不稳定警告(NaN/Inf)
- [ ] 单样本 split 能正确报错

### P1 完成标准:
- [ ] 缓存命中率 >60%
- [ ] 训练速度提升 >20%
- [ ] 代码中 magic number <5 个

### P2 完成标准:
- [ ] 生成至少 4 种可视化图表
- [ ] 课程学习收敛速度提升 >10%
- [ ] 完成温度参数消融实验(6 组)

---

## 📝 代码审查 Checklist

提交前自检:
- [ ] 类型提示覆盖率 >90%
- [ ] 关键函数有 docstring
- [ ] 无硬编码路径
- [ ] 配置默认值合理
- [ ] 错误信息清晰
- [ ] 添加单元测试
- [ ] 更新 README 文档

---

**文档创建时间:** 2026-06-08  
**预计完成时间:** 2026-06-22 (2 周)  
**责任人:** 待分配
