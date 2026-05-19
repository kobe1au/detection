# 实验结果汇总表 — 快速参考

## 表 1: 创新点一内部消融（TPCL + IGA）

| ID | 方法 | TPCL | IGA | Temporal Selection | Latest Calib | 预期 AUT(F1) |
|:---:|---|:---:|:---:|:---:|:---:|---:|
| 1 | ERM | ✗ | ✗ | ✗ | ✗ | baseline |
| 2 | ERM + Temporal | ✗ | ✗ | ✓ | ✓ | ↑ |
| 3 | +TPCL | ✓ | ✗ | ✓ | ✓ | ↑↑ |
| 4 | +IGA | ✓ | ✓ | ✓ | ✓ | ↑↑↑ |

> 预期：TPCL 和 IGA 各自提升，叠加最高。TPCL 贡献更大（跨年份表征对齐），IGA 贡献辅助（优化方向对齐）。

## 表 2: 创新点二消融（Alignment Bias）

| ID | 方法 | TPCL+IGA | Align Bias | Context Refine | 预期 AUT(F1) |
|:---:|---|:---:|:---:|:---:|---:|
| 5 | Ours_no_align | ✓ | ✗ | ✗ | baseline+ |
| 6 | Ours_align | ✓ | ✓ (bonus=1.0) | ✓ (scale=0.35) | ↑↑ |

> 预期：Alignment Bias 在 temporal 基座上进一步提升，尤其在年份间结构变化较大时更明显。

## 表 3: 创新点三消融（Quality/Drift Gate）

| ID | 方法 | Align | Quality Gate | Drift Gate | 预期 AUT(F1) |
|:---:|---|:---:|:---:|:---:|---:|
| 7 | Ours_gate_neutral | ✓ | ✗ | ✗ | baseline++ |
| 8 | Ours_gate_quality | ✓ | ✓ | ✗ | ↑ |
| 9 | Ours_gate_drift | ✓ | ✗ | ✓ | ↑ |
| 10 | Ours_gate_full | ✓ | ✓ | ✓ | ↑↑ |

> 预期：quality 和 drift 各自提供互补信号，full gate 效果最好。若单独已接近 full，说明两者存在冗余，可选择性简化。

## 表 4: 基线对比

| ID | 方法 | 单/多模态 | Temporal | 预期相对 Ours |
|:---:|---|:---:|:---:|---:|
| 11 | API Only | 单 | ✓ | 最低 |
| 12 | Graph Only | 单 | ✓ | 较低 |
| 13 | Concat | 多 | ✓ | 中等 |
| 14 | Cross-Attention | 多 | ✓ | 中等偏上 |
| 15 | **Ours Full** | 多 | ✓ | 最高 |

## 表 5: 主要对比维度总结

| 维度 | 解决的问题 | 核心机制 | 位置 |
|---|---|---|---|
| 创新点一 | 跨时间分布漂移 | TPCL + IGA + Temporal Selection + Latest Calib | [fusion/losses.py:190](fusion/losses.py#L190), [fusion/losses.py:270](fusion/losses.py#L270), [fusion/train.py:540](fusion/train.py#L540) |
| 创新点二 | API-Graph 结构不对齐 | Alignment Bias (bonus-only) + Context Refinement | [fusion/model.py:951](fusion/model.py#L951), [fusion/model.py:629](fusion/model.py#L629) |
| 创新点三 | 模态质量动态变化 | Quality Gate + Drift Score + Tri-Branch | [fusion/model.py:587](fusion/model.py#L587), [fusion/modules.py:30](fusion/modules.py#L30) |

## 预期实验结论

1. **TPCL 比 IGA 更关键**：TPCL 直接约束表征空间，对跨年份泛化贡献更大。
2. **Alignment Bias 在 temporal 基座上仍然有效**：说明结构先验和时间鲁棒性是正交增强。
3. **Quality 和 Drift 互补**：若两者单独提升相近，可考虑只保留 drift（计算更简单）。
4. **Ours Full 应在所有指标上领先**：AUT(F1)、Latest F1、Worst F1、AURC、Robustness Suite。